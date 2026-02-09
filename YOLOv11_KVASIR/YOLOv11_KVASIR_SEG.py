import os
import shutil
import yaml
import numpy as np
import cv2
import glob
import subprocess
import sys

# Function to install packages if missing
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Checking dependencies...")
try:
    import gdown
    from ultralytics import YOLO
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
except ImportError:
    print("Installing missing dependencies...")
    install("ultralytics")
    install("gdown")
    install("scikit-learn")
    install("opencv-python-headless")
    # Re-import after installation
    import gdown
    from ultralytics import YOLO
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# Use current directory for remote execution
ROOT_DIR = os.getcwd()
YOLO_ROOT = os.path.join(ROOT_DIR, "yolo_kvasir_dataset")
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "data/images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "data/masks")
KVASIR_DIR = os.path.join(ROOT_DIR, "Kvasir-SEG")
KVASIR_IMG_DIR = os.path.join(KVASIR_DIR, "images")
KVASIR_MASK_DIR = os.path.join(KVASIR_DIR, "masks")

print(f"Working Directory: {ROOT_DIR}")

# --- 1. Download Primary Training Data (from GDrive) ---
print("\n--- Step 1: Training Data ---")
file_id = '1Bv7zmjJVvE7uQbbjwRa-3qn-BI8h99KI'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'data.zip'

if not os.path.exists(output):
    print("Downloading data.zip...")
    gdown.download(url, output, quiet=False)

if os.path.exists(output) and not os.path.exists(os.path.join(DATA_DIR, "data")):
    print("Unzipping Training Data...")
    import zipfile
    print("Unzipping Training Data...")
    with zipfile.ZipFile("data.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    print("✅ Training Data ready.")
else:
    print("✅ Training Data already extracted.")

# --- 2. Download KVASIR-SEG Validation Data ---
print("\n--- Step 2: Validation Data (KVASIR-SEG) ---")
kvasir_url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
kvasir_output = "kvasir-seg.zip"

if not os.path.exists(kvasir_output):
    print(f"Downloading Kvasir-SEG from {kvasir_url}...")
    import urllib.request
    import ssl
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(kvasir_url, context=ctx) as response, open(kvasir_output, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        if os.path.getsize(kvasir_output) == 0:
             print("❌ Download failed: File is empty.")
             os.remove(kvasir_output)
        else:
             print("✅ Download successful.")
    except Exception as e:
        print(f"❌ Download failed: {e}")
    
if os.path.exists(kvasir_output) and not os.path.exists(KVASIR_DIR):
    print("Unzipping Kvasir-SEG...")
    import zipfile
    print("Unzipping Kvasir-SEG...")
    with zipfile.ZipFile(kvasir_output, 'r') as zip_ref:
        zip_ref.extractall(ROOT_DIR)
    print("✅ Kvasir-SEG ready.")
else:
    print("✅ Kvasir-SEG already extracted.")

# --- 3. Preprocessing Functions ---
def mask_to_yolo_polygon(mask_path):
    """Converts a binary mask to YOLO polygon format."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    
    # Normalize if needed (0-1 to 0-255)
    if np.max(mask) <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = mask.shape
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / w)
                polygon.append(y / h)
            if len(polygon) > 4:
                polygons.append(polygon)
    return polygons

def prepare_yolo_dataset():
    """Prepares the YOLO dataset structure with Train (all files) and Val (KVASIR)."""
    print("\n--- Step 3: Preparing YOLO Dataset ---")
    
    import random
    random.seed(42) # Ensure reproducibility

    # Clean/Create Dirs
    if os.path.exists(YOLO_ROOT):
        print(f"Removing existing {YOLO_ROOT}...")
        subprocess.run(["rm", "-rf", YOLO_ROOT])
    
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{YOLO_ROOT}/{split}/images", exist_ok=True)
        os.makedirs(f"{YOLO_ROOT}/{split}/labels", exist_ok=True)
    
    # --- Process PRIMARY Data (Gdrive) -> Train (80%) + Val (20%) ---
    if os.path.exists(TRAIN_IMG_DIR):
        all_files = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * 0.8)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        print(f"Primary Dataset: {len(all_files)} images")
        print(f"  - Train (80%): {len(train_files)} images")
        print(f"  - Val   (20%): {len(val_files)} images")
        
        # Helper to process a list of files into a specific split
        def process_split(files, split_name):
            for f in tqdm(files, desc=f"Processing {split_name}"):
                base_name = os.path.splitext(f)[0]
                
                # Copy Image
                shutil.copy(os.path.join(TRAIN_IMG_DIR, f), os.path.join(YOLO_ROOT, split_name, 'images', f))
                
                # Find Mask
                mask_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    try_path = os.path.join(TRAIN_MASK_DIR, base_name + ext)
                    if os.path.exists(try_path):
                        mask_path = try_path
                        break
                
                # Create Label
                label_path = os.path.join(YOLO_ROOT, split_name, 'labels', base_name + ".txt")
                if mask_path:
                    polygons = mask_to_yolo_polygon(mask_path)
                    with open(label_path, 'w') as out:
                        for poly in polygons:
                            out.write("0 " + " ".join(map(str, poly)) + "\n")
                else:
                    open(label_path, 'w').close()

        process_split(train_files, 'train')
        process_split(val_files, 'val')
        
    else:
        print(f"⚠️ Training directory not found: {TRAIN_IMG_DIR}")

    # --- Process KVASIR Data -> Test (External) ---
    if os.path.exists(KVASIR_IMG_DIR):
        kvasir_files = sorted([f for f in os.listdir(KVASIR_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f"Processing {len(kvasir_files)} KVASIR images -> Test Set...")
        
        for f in tqdm(kvasir_files, desc="Processing Test (KVASIR)"):
            base_name = os.path.splitext(f)[0]
            
            # Copy Image
            shutil.copy(os.path.join(KVASIR_IMG_DIR, f), os.path.join(YOLO_ROOT, 'test', 'images', f))
            
            # Find Mask
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                try_path = os.path.join(KVASIR_MASK_DIR, base_name + ext)
                if os.path.exists(try_path):
                    mask_path = try_path
                    break
            
            label_path = os.path.join(YOLO_ROOT, 'test', 'labels', base_name + ".txt")
            if mask_path:
                polygons = mask_to_yolo_polygon(mask_path)
                with open(label_path, 'w') as out:
                    for poly in polygons:
                        out.write("0 " + " ".join(map(str, poly)) + "\n")
            else:
                 open(label_path, 'w').close()
    else:
        print(f"⚠️ KVASIR directory not found: {KVASIR_IMG_DIR}")

    # Create YAML
    data_config = {
        'path': YOLO_ROOT,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {0: 'polyp'}
    }
    with open(f"{YOLO_ROOT}/data.yaml", 'w') as f:
        yaml.dump(data_config, f)
    
    print("✅ Dataset Preparation Complete.")

prepare_yolo_dataset()

# --- 4. Train YOLOv11 Model ---
print("\n--- Step 4: Training YOLOv11 ---")
# Check if model exists locally or needs download. YOLO class handles download.
model = YOLO('yolo11m-seg.pt')

results = model.train(
    data=f"{YOLO_ROOT}/data.yaml",
    epochs=100,
    imgsz=256,
    batch=16,
    project=os.path.join(ROOT_DIR, "yolo_kvasir_results"),
    name="train_all_val_kvasir",
    augment=True,
    exist_ok=True,
    verbose=True
)

# --- 5. External Validation on KVASIR-SEG ---
print("\n--- Step 5: Detailed Validation on KVASIR-SEG ---")

best_model_path = os.path.join(ROOT_DIR, "yolo_kvasir_results/train_all_val_kvasir/weights/best.pt")
if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    val_model = YOLO(best_model_path)
else:
    print("⚠️ Best model not found, using current model weights.")
    val_model = model

scores = {"dice": [], "iou": [], "prec": [], "rec": [], "acc": []}

val_images_path = os.path.join(YOLO_ROOT, "test", "images")
results = val_model.predict(source=val_images_path, imgsz=256, conf=0.25, verbose=False, retina_masks=True)

for result in tqdm(results, desc="Evaluating"):
    filename = os.path.basename(result.path)
    base_name = os.path.splitext(filename)[0]
    
    # Find GT Mask
    gt_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        p = os.path.join(KVASIR_MASK_DIR, base_name + ext)
        if os.path.exists(p): 
            gt_path = p
            break
    
    if not gt_path: continue
    
    # Process Pred
    h, w = result.orig_shape
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks:
        for m in result.masks.data:
            m_np = m.cpu().numpy()
            m_resized = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_mask = np.maximum(pred_mask, m_resized)
    pred_bin = (pred_mask > 0.5).astype(np.uint8).flatten()
    
    # Process GT
    gt = np.array(Image.open(gt_path).convert("L"))
    gt_bin = (gt > 127).astype(np.uint8).flatten()
    
    # Metrics
    scores["dice"].append(f1_score(gt_bin, pred_bin, zero_division=1))
    scores["iou"].append(jaccard_score(gt_bin, pred_bin, zero_division=1))
    scores["prec"].append(precision_score(gt_bin, pred_bin, zero_division=1))
    scores["rec"].append(recall_score(gt_bin, pred_bin, zero_division=1))
    scores["acc"].append(accuracy_score(gt_bin, pred_bin))

avg_scores = {k: np.mean(v) for k, v in scores.items()}
print(f"\n{'='*40}")
print(f"🔹 KVASIR-SEG Validation Results 🔹")
print(f"{'='*40}")
print(f"Dice Coefficient : {avg_scores['dice']:.4f}")
print(f"IoU (Jaccard)    : {avg_scores['iou']:.4f}")
print(f"Precision        : {avg_scores['prec']:.4f}")
print(f"Recall           : {avg_scores['rec']:.4f}")
print(f"Accuracy         : {avg_scores['acc']:.4f}")
print(f"{'='*40}")
