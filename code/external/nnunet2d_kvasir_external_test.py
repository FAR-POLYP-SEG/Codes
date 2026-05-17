import os
import shutil
import json
import numpy as np
import SimpleITK as sitk
import gdown
import sys
import zipfile
import urllib.request
import ssl
import subprocess
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# --- Configuration ---
SEED = 42
DATASET_ID = 501
TASK_NAME = f"Dataset{DATASET_ID:03d}_Polyp"

# Environment Setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(ROOT_DIR)
NNUNET_RAW = os.path.join(ROOT_DIR, "nnUNet_raw")
NNUNET_PREPROCESSED = os.path.join(ROOT_DIR, "nnUNet_preprocessed")
NNUNET_RESULTS = os.path.join(ROOT_DIR, "nnUNet_results")

os.environ['nnUNet_raw'] = NNUNET_RAW
os.environ['nnUNet_preprocessed'] = NNUNET_PREPROCESSED
os.environ['nnUNet_results'] = NNUNET_RESULTS

# Paths
DATA_DIR = os.path.join(ROOT_DIR, "data/data")
IMAGES_TR = os.path.join(NNUNET_RAW, TASK_NAME, "imagesTr")
LABELS_TR = os.path.join(NNUNET_RAW, TASK_NAME, "labelsTr")
IMAGES_TS = os.path.join(NNUNET_RAW, TASK_NAME, "imagesTs")
LABELS_TS = os.path.join(NNUNET_RAW, TASK_NAME, "labelsTs")  # For Eval GT

for p in [IMAGES_TR, LABELS_TR, IMAGES_TS, LABELS_TS, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
    os.makedirs(p, exist_ok=True)

# --- 1. Download Data ---
print("\n--- Step 1: Downloading Data ---")

# Primary Data
file_id = '1Bv7zmjJVvE7uQbbjwRa-3qn-BI8h99KI'
url = f'https://drive.google.com/uc?id={file_id}'
output = os.path.join(ROOT_DIR, "data.zip")

if not os.path.exists(output):
    print("Downloading data.zip...")
    gdown.download(url, output, quiet=False)

if os.path.exists(output) and not os.path.exists(os.path.join(ROOT_DIR, "data")):
    print("Unzipping Training Data...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(ROOT_DIR, "data"))
    print("✅ Training Data ready.")

# KVASIR-SEG
kvasir_url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
kvasir_output = os.path.join(ROOT_DIR, "kvasir-seg.zip")

if not os.path.exists(kvasir_output):
    print(f"Downloading Kvasir-SEG from {kvasir_url}...")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(kvasir_url, context=ctx) as response, open(kvasir_output, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    except Exception as e:
        print(f"❌ Download failed: {e}")

if os.path.exists(kvasir_output) and not os.path.exists(os.path.join(ROOT_DIR, "Kvasir-SEG")):
    print("Unzipping Kvasir-SEG...")
    with zipfile.ZipFile(kvasir_output, 'r') as zip_ref:
        zip_ref.extractall(ROOT_DIR)
    print("✅ Kvasir-SEG ready.")

# --- 2. Convert to NIfTI ---
print("\n--- Step 2: Converting to NIfTI ---")

def convert_to_nifti(src_img_dir, src_mask_dir, dest_img_dir, dest_lbl_dir, file_list, is_test=False):
    processed_files = []
    
    for f in file_list:
        img_path = os.path.join(src_img_dir, f)
        base_name = os.path.splitext(f)[0]
        
        # Find Mask
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            try_path = os.path.join(src_mask_dir, base_name + ext)
            if os.path.exists(try_path): 
                mask_path = try_path
                break
        
        # Use image name as fallback if strict name match fails (common in these datasets)
        if not mask_path:
             try_path = os.path.join(src_mask_dir, f)
             if os.path.exists(try_path): mask_path = try_path

        if not mask_path: continue

        # Load & Convert
        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")
        
        # Resize to 256x256 slightly speeds up and standardizes things (optional but good for consistency)
        # nnU-Net handles spacing provided we set it. Let's keep original size for nnU-Net as it adapts.
        
        img_arr = np.array(img_pil) # (H, W, 3)
        mask_arr = np.array(mask_pil) # (H, W)
        
        # Threshold Mask
        mask_arr = (mask_arr > 127).astype(np.uint8)
        
        # Transpose to (X, Y, C) -> SimpleITK expects (X, Y) order but numpy is (Y, X)
        # Actually SimpleITK GetImageFromArray takes (Z, Y, X) or (Y, X).
        # We need to be careful. nnU-Net 2D expects (C, Y, X)
        
        # For RGB images, nnU-Net handles them as 3 separate channels.
        # We need to save as 3D nifti or 4D?
        # Standard nnU-Net for RGB: Save 3 separate files _0000, _0001, _0002 OR
        # Convert to Grayscale if color doesn't matter much? 
        # Previous models used RGB.
        # Let's use RGB. We need R, G, B channels as modalities 0, 1, 2.
        
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        
        img_r = sitk.GetImageFromArray(r)
        img_g = sitk.GetImageFromArray(g)
        img_b = sitk.GetImageFromArray(b)
        mask_obj = sitk.GetImageFromArray(mask_arr)
        
        for obj in [img_r, img_g, img_b, mask_obj]:
            obj.SetSpacing((1.0, 1.0))
            obj.SetOrigin((0.0, 0.0))
            obj.SetDirection((1.0, 0.0, 0.0, 1.0))
            
        # Save R, G, B channels
        sitk.WriteImage(img_r, os.path.join(dest_img_dir, f"{base_name}_0000.nii.gz"))
        sitk.WriteImage(img_g, os.path.join(dest_img_dir, f"{base_name}_0001.nii.gz"))
        sitk.WriteImage(img_b, os.path.join(dest_img_dir, f"{base_name}_0002.nii.gz"))
        
        # Save Mask
        sitk.WriteImage(mask_obj, os.path.join(dest_lbl_dir, f"{base_name}.nii.gz"))
        
        processed_files.append(base_name)
        
    return processed_files

# --- Primary Dataset ---
primary_img_dir = os.path.join(DATA_DIR, "images")
primary_mask_dir = os.path.join(DATA_DIR, "masks")
all_primary_files = sorted([f for f in os.listdir(primary_img_dir) if f.endswith('.jpg') or f.endswith('.png')])

print(f"Converting {len(all_primary_files)} Primary images...")
train_ids = convert_to_nifti(primary_img_dir, primary_mask_dir, IMAGES_TR, LABELS_TR, all_primary_files)

# --- KVASIR Dataset ---
kvasir_img_dir = os.path.join(ROOT_DIR, "Kvasir-SEG/images")
kvasir_mask_dir = os.path.join(ROOT_DIR, "Kvasir-SEG/masks")
all_kvasir_files = sorted([f for f in os.listdir(kvasir_img_dir) if f.endswith('.jpg') or f.endswith('.png')])

print(f"Converting {len(all_kvasir_files)} KVASIR images...")
test_ids = convert_to_nifti(kvasir_img_dir, kvasir_mask_dir, IMAGES_TS, LABELS_TS, all_kvasir_files, is_test=True)


# --- 3. Create dataset.json ---
print("\n--- Step 3: Creating dataset.json ---")
json_dict = {
    "channel_names": {
        "0": "R",
        "1": "G",
        "2": "B"
    },
    "labels": {
        "background": 0,
        "polyp": 1
    },
    "numTraining": len(train_ids),
    "file_ending": ".nii.gz"
}
with open(os.path.join(NNUNET_RAW, TASK_NAME, "dataset.json"), 'w') as f:
    json.dump(json_dict, f)


# --- 4. Plan & Preprocess ---
print("\n--- Step 4: Plan & Preprocess (this may take a while) ---")
# Only run if not already done to save time on re-runs
if not os.path.exists(os.path.join(NNUNET_PREPROCESSED, TASK_NAME, "nnUNetPlans.json")):
    cmd = f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d --verify_dataset_integrity"
    subprocess.run(cmd, shell=True, check=True)
else:
    print("Preprocessed data found, skipping...")


# --- 5. Custom Split (80/20) ---
print("\n--- Step 5: Enforcing 80/20 Split ---")
# Split Primary IDs
random.seed(SEED)
random.shuffle(train_ids)
split_idx = int(len(train_ids) * 0.8)
train_split = train_ids[:split_idx]
val_split = train_ids[split_idx:]

splits = [
    {
        "train": train_split,
        "val": val_split
    }
]

split_file = os.path.join(NNUNET_PREPROCESSED, TASK_NAME, "splits_final.json")
with open(split_file, 'w') as f:
    json.dump(splits, f)

print(f"Split created: {len(train_split)} Train, {len(val_split)} Val")


# --- 6. Train (1000 Epochs) ---
print("\n--- Step 6: Training (Default 1000 Epochs) ---")
print("Starts training... output will be logged to training_nnunet.log")
# We run this command and wait for it finish.
# Since user asked for default epochs (1000), this will take time.
# We explicitly use the split file we created on fold 0.
cmd_train = f"nnUNetv2_train {DATASET_ID} 2d 0"
subprocess.run(cmd_train, shell=True, check=True) # This is blocking


# --- 7. Inference & Evaluation ---
print("\n--- Step 7: Inference on KVASIR ---")
output_dir = os.path.join(NNUNET_RESULTS, TASK_NAME, "inference_kvasir")
cmd_predict = f"nnUNetv2_predict -i {IMAGES_TS} -o {output_dir} -d {DATASET_ID} -c 2d -f 0"
subprocess.run(cmd_predict, shell=True, check=True)

print("\n--- Step 8: Calculate Metrics ---")
scores = {"dice": [], "iou": [], "prec": [], "rec": [], "acc": []}

for case_id in test_ids:
    pred_path = os.path.join(output_dir, f"{case_id}.nii.gz")
    gt_path = os.path.join(LABELS_TS, f"{case_id}.nii.gz")
    
    if not os.path.exists(pred_path): continue
    
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)).flatten()
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).flatten()
    
    # Binarize
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    scores["dice"].append(f1_score(gt, pred, zero_division=1))
    scores["iou"].append(jaccard_score(gt, pred, zero_division=1))
    scores["prec"].append(precision_score(gt, pred, zero_division=1))
    scores["rec"].append(recall_score(gt, pred, zero_division=1))
    scores["acc"].append(accuracy_score(gt, pred))

avg_scores = {k: np.mean(v) for k, v in scores.items()}

print(f"\n{'='*40}")
print(f"🔹 KVASIR-SEG External Test Results (nnU-Net 2D) 🔹")
print(f"{'='*40}")
print(f"Dice Coefficient : {avg_scores['dice']:.4f}")
print(f"IoU (Jaccard)    : {avg_scores['iou']:.4f}")
print(f"Precision        : {avg_scores['prec']:.4f}")
print(f"Recall           : {avg_scores['rec']:.4f}")
print(f"Accuracy         : {avg_scores['acc']:.4f}")
print(f"{'='*40}")

with open("nnunet_metrics.txt", "w") as f:
    f.write(str(avg_scores))
