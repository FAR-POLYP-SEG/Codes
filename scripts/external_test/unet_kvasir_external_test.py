import os
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import gdown
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import cv2
import zipfile
import shutil
import urllib.request
import ssl
import sys
import subprocess
from PIL import Image

# --- Configuration ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data") # Primary dataset root
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "data/images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "data/masks")

KVASIR_DIR = os.path.join(ROOT_DIR, "Kvasir-SEG")
KVASIR_IMG_DIR = os.path.join(KVASIR_DIR, "images")
KVASIR_MASK_DIR = os.path.join(KVASIR_DIR, "masks")

RESULTS_DIR = os.path.join(ROOT_DIR, "unet_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. Data Download ---
print("\n--- Step 1: Downloading Data ---")

# Primary Data
file_id = '1Bv7zmjJVvE7uQbbjwRa-3qn-BI8h99KI'
url = f'https://drive.google.com/uc?id={file_id}'
output = os.path.join(ROOT_DIR, "data.zip")

if not os.path.exists(output):
    print("Downloading data.zip...")
    gdown.download(url, output, quiet=False)

if os.path.exists(output) and not os.path.exists(os.path.join(DATA_DIR, "data")):
    print("Unzipping Training Data...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(ROOT_DIR, "data"))
    print("✅ Training Data ready.")
else:
    print("✅ Training Data already extracted.")

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
            
        if os.path.getsize(kvasir_output) == 0:
             print("❌ Download failed: File is empty.")
             os.remove(kvasir_output)
        else:
             print("✅ Download successful.")
    except Exception as e:
        print(f"❌ Download failed: {e}")

if os.path.exists(kvasir_output) and not os.path.exists(KVASIR_DIR):
    print("Unzipping Kvasir-SEG...")
    with zipfile.ZipFile(kvasir_output, 'r') as zip_ref:
        zip_ref.extractall(ROOT_DIR)
    print("✅ Kvasir-SEG ready.")
else:
    print("✅ Kvasir-SEG already extracted.")


# --- 2. Dataset & transforms ---
# Augmentations
train_transform = A.Compose([
    A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    A.ToFloat(max_value=255.0),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

val_transform = A.Compose([
    A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR),
    A.ToFloat(max_value=255.0),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = file_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Determine mask extension logic
        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            if os.path.exists(os.path.join(self.mask_dir, base_name + ext)):
                 mask_path = os.path.join(self.mask_dir, base_name + ext)
                 break
        
        # Fallback if mask not found
        if mask_path is None:
             mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        if mask_path and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
        else:
            # Create empty mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0).float()
            mask = (mask > 0.5).float()

        return image, mask

# --- 3. Split Data ---
print("\n--- Step 2: Splitting Data ---")
# Primary Data Splitting
if os.path.exists(TRAIN_IMG_DIR):
    all_files = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Primary Dataset: {len(all_files)} images")
    print(f"  - Train (80%): {len(train_files)} images")
    print(f"  - Val   (20%): {len(val_files)} images")
else:
    print("❌ Primary dataset directory not found!")
    sys.exit(1)

# KVASIR (Test)
if os.path.exists(KVASIR_IMG_DIR):
    test_files = sorted([f for f in os.listdir(KVASIR_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Test Dataset (KVASIR): {len(test_files)} images")
    
    # Verify masks exist for test files to avoid errors
    valid_test_files = []
    for f in test_files:
        base_name = os.path.splitext(f)[0]
        has_mask = False
        for ext in ['.png', '.jpg', '.jpeg']:
             if os.path.exists(os.path.join(KVASIR_MASK_DIR, base_name + ext)):
                 has_mask = True
                 break
        if has_mask:
            valid_test_files.append(f)
            
    print(f"  - Valid Test Pairs: {len(valid_test_files)} images")
    test_files = valid_test_files

else:
    print("❌ KVASIR dataset directory not found!")
    test_files = []

# DataLoaders
train_dataset = CustomDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_files, transform=train_transform)
val_dataset = CustomDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, val_files, transform=val_transform)
test_dataset = CustomDataset(KVASIR_IMG_DIR, KVASIR_MASK_DIR, test_files, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- 4. Model Architecture (Unet via SMP) ---
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(device)

loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- 5. Training & Validation Functions ---
def validate(model, loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)
            val_loss += loss.item()
    return val_loss / len(loader)

def train_and_validate():
    print("\n--- Step 3: Starting Training (Unet ResNet34) ---")
    
    best_loss = float('inf')
    EPOCHS = 100
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Validation
        val_loss = validate(model, val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
            print("  --> New Best Model Saved!")
            
    print("✅ Training Complete.")

# --- 6. Final Evaluation on KVASIR ---
def evaluate_test_set():
    print("\n--- Step 4: Final Evaluation on KVASIR (Test Set) ---")
    
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pth")))
    model.eval()
    
    scores = {"dice": [], "iou": [], "prec": [], "rec": [], "acc": []}
    
    with torch.no_grad():
        for image, true_mask in test_loader:
            input_batch = image.to(device)
            logits = model(input_batch)
            probs = logits.sigmoid()
            pred_mask = (probs > 0.5).float()
            
            p = pred_mask.cpu().squeeze().numpy().flatten().astype(np.uint8)
            g = true_mask.cpu().squeeze().numpy().flatten().astype(np.uint8)
            
            scores["dice"].append(f1_score(g, p, zero_division=1))
            scores["iou"].append(jaccard_score(g, p, zero_division=1))
            scores["prec"].append(precision_score(g, p, zero_division=1))
            scores["rec"].append(recall_score(g, p, zero_division=1))
            scores["acc"].append(accuracy_score(g, p))
            
    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    
    print(f"\n{'='*40}")
    print(f"🔹 KVASIR-SEG External Test Results (Unet) 🔹")
    print(f"{'='*40}")
    print(f"Dice Coefficient : {avg_scores['dice']:.4f}")
    print(f"IoU (Jaccard)    : {avg_scores['iou']:.4f}")
    print(f"Precision        : {avg_scores['prec']:.4f}")
    print(f"Recall           : {avg_scores['rec']:.4f}")
    print(f"Accuracy         : {avg_scores['acc']:.4f}")
    print(f"{'='*40}")
    
    # Save results to a file for easy reading
    with open("unet_kvasir_metrics.txt", "w") as f:
        f.write(str(avg_scores))

if __name__ == "__main__":
    train_and_validate()
    evaluate_test_set()
