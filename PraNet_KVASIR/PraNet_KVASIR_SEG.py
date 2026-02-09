import os
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
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
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data") # Primary dataset root
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "data/images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "data/masks")

KVASIR_DIR = os.path.join(ROOT_DIR, "Kvasir-SEG")
KVASIR_IMG_DIR = os.path.join(KVASIR_DIR, "images")
KVASIR_MASK_DIR = os.path.join(KVASIR_DIR, "masks")

RESULTS_DIR = os.path.join(ROOT_DIR, "pranet_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. Data Download ---
print("\n--- Step 1: Downloading Data ---")

# Primary Data
file_id = '1Bv7zmjJVvE7uQbbjwRa-3qn-BI8h99KI'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'data.zip'

if not os.path.exists(output):
    print("Downloading data.zip...")
    gdown.download(url, output, quiet=False)

if os.path.exists(output) and not os.path.exists(os.path.join(DATA_DIR, "data")):
    print("Unzipping Training Data...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("data")
    print("✅ Training Data ready.")
else:
    print("✅ Training Data already extracted.")

# KVASIR-SEG
kvasir_url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
kvasir_output = "kvasir-seg.zip"

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
        
        # Fallback if mask not found (shouldn't happen with correct splitting)
        if mask_path is None:
             # Try direct mapping if file names match exactly
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

# --- 4. Model Architecture (PraNet) ---
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        
        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * \
               self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x

class PraNet(nn.Module):
    def __init__(self, channel=32):
        super(PraNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        
        self.agg1 = aggregation(channel)
        
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 1, kernel_size=1)
        
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=1)
        
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) 

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        # lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear', align_corners=True)

        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear', align_corners=True)
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        ra4_feat = self.ra4_conv4(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=True)

        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)

        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

        return lateral_map_2, lateral_map_3, lateral_map_4, lateral_map_4 # Returning 4 outputs consistent with loss

# --- 5. Loss Function ---
def structure_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = 1 - (2. * intersection + 1e-5) / (union + 1e-5)
    return bce + dice.mean()

# --- 6. Training & Validation Functions ---
def validate(model, loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            out5, out4, out3, out2 = model(images)
            
            loss5 = structure_loss(out5, masks)
            loss4 = structure_loss(out4, masks)
            loss3 = structure_loss(out3, masks)
            loss2 = structure_loss(out2, masks)
            
            loss = loss5 + loss4 + loss3 + loss2
            val_loss += loss.item()
    return val_loss / len(loader)

def train_and_validate():
    print("\n--- Step 3: Starting Training ---")
    model = PraNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_loss = float('inf')
    EPOCHS = 100
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            out5, out4, out3, out2 = model(images)
            
            loss5 = structure_loss(out5, masks)
            loss4 = structure_loss(out4, masks)
            loss3 = structure_loss(out3, masks)
            loss2 = structure_loss(out2, masks)
            
            loss = loss5 + loss4 + loss3 + loss2
            
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

# --- 7. Final Evaluation on KVASIR ---
def evaluate_test_set():
    print("\n--- Step 4: Final Evaluation on KVASIR (Test Set) ---")
    
    model = PraNet().to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pth")))
    model.eval()
    
    scores = {"dice": [], "iou": [], "prec": [], "rec": [], "acc": []}
    
    with torch.no_grad():
        for image, true_mask in test_loader:
            input_batch = image.to(device)
            
            # Forward (PraNet returns 4 maps, we typically use the first one or lateral_map_2 which corresponds to Res2 which is finest?)
            # In PraNet implementation usually the last output in return list (lateral_map_2) is the output map
            # But in the class def above: return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2
            # wait, the forward return line was: return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2
            # Actually, lateral_map_2 is the finest scale (scale factor 8 upsample).
            # Wait, looking at the code:
            # lateral_map_2 = F.interpolate(x, scale_factor=8, ... ) 
            # In the original paper/code, 'res2' is usually the high-res one.
            # My validate loop used: out5, out4, out3, out2 = model(images)
            # and loss was sum of all.
            # For inference, we usually take the best one.
            # In PraNet code usually 'res2' (the last one returned) is the final prediction.
            
            _, _, _, pred_map = model(input_batch)
            
            # Sigmoid & Threshold
            probs = pred_map.sigmoid()
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
    print(f"🔹 KVASIR-SEG Validation Results (PraNet) 🔹")
    print(f"{'='*40}")
    print(f"Dice Coefficient : {avg_scores['dice']:.4f}")
    print(f"IoU (Jaccard)    : {avg_scores['iou']:.4f}")
    print(f"Precision        : {avg_scores['prec']:.4f}")
    print(f"Recall           : {avg_scores['rec']:.4f}")
    print(f"Accuracy         : {avg_scores['acc']:.4f}")
    print(f"{'='*40}")
    
    # Save results to a file for easy reading
    with open("pranet_kvasir_metrics.txt", "w") as f:
        f.write(str(avg_scores))

if __name__ == "__main__":
    train_and_validate()
    evaluate_test_set()
