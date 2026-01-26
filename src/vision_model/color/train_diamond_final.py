# ------------------------------------------------------------
# DIAMOND COLOR - FINAL PRODUCTION VERSION
# Features: Progress Bar + Weighted Sampler + Resizing Fix
# ------------------------------------------------------------

import os
import time
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm  # <--- Progress Bar Library

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

# ------------------------------
# Dependencies Check
# ------------------------------
try:
    import kagglehub
except ImportError:
    raise ImportError("Please run: pip install kagglehub")

try:
    from skimage import color as skcolor
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: skimage not found. Install 'scikit-image' for better color accuracy.")

# ------------------------------
# Config
# ------------------------------
@dataclass
class CFG:
    dataset_handle: str = "aayushpurswani/diamond-images-dataset"
    seed: int = 42
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    num_workers: int = 0
    lr: float = 2e-4
    weight_decay: float = 1e-2
    out_dir_name: str = "_final_model_out"
    model_name: str = "efficientnet_v2_s" 

TIER7_ORDER = [
    "Premium_White",       # D-F
    "Near_Colorless_High", # G-H
    "Near_Colorless_Low",  # I-J
    "Faint_Yellow",        # K-L
    "Very_Light_Yellow",   # M-N
    "Light_Yellow",        # O-R
    "Yellow_LowEnd",       # S-Z
]

# ------------------------------
# Utils
# ------------------------------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def map_color_to_tier7(color_str: str) -> str:
    c = str(color_str).strip().upper()
    if c in ["D", "E", "F"]: return "Premium_White"
    if c in ["G", "H"]: return "Near_Colorless_High"
    if c in ["I", "J"]: return "Near_Colorless_Low"
    if c in ["K", "L"]: return "Faint_Yellow"
    if c in ["M", "N"]: return "Very_Light_Yellow"
    if c in ["O-P", "Q-R", "O", "P", "Q", "R"]: return "Light_Yellow"
    if c in ["S-T", "U-V", "W-X", "Y-Z", "S", "T", "U", "V", "W", "X", "Y", "Z"]: return "Yellow_LowEnd"
    return "UNKNOWN"

def shades_of_gray_wb(img_rgb: np.ndarray, p: int = 6, eps: float = 1e-6) -> np.ndarray:
    """Removes lighting tint."""
    img = np.clip(img_rgb, 0.0, 1.0)
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p) + eps
    illum = illum / (np.mean(illum) + eps)
    out = img / illum[None, None, :]
    return np.clip(out, 0.0, 1.0)

# ------------------------------
# File Mapping Logic
# ------------------------------
def list_all_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

def build_image_indexes(image_paths: List[Path]):
    by_base = {p.stem: p for p in image_paths}
    by_base_lower = {p.stem.lower(): p for p in image_paths}
    return {"base": by_base, "lower": by_base_lower}

def smart_map_images(df: pd.DataFrame, root_dir: Path) -> pd.DataFrame:
    imgs = list_all_images(root_dir)
    idx = build_image_indexes(imgs)
    
    candidates = [c for c in df.columns if any(x in c for x in ['image', 'path', 'file', 'id'])]
    if not candidates: candidates = df.columns
    
    best_col, best_matches, best_paths = None, 0, []
    
    for col in candidates:
        matches = 0
        paths = []
        for val in df[col].astype(str):
            fname = Path(val).stem
            if fname in idx["base"]:
                paths.append(str(idx["base"][fname]))
                matches += 1
            elif fname.lower() in idx["lower"]:
                paths.append(str(idx["lower"][fname.lower()]))
                matches += 1
            else:
                paths.append(None)
        
        if matches > best_matches:
            best_matches = matches
            best_col = col
            best_paths = paths
            
    if best_matches < len(df) * 0.1:
        log("!!! CRITICAL: Could not map CSV to Images automatically.")
        return df
    
    df["image_path"] = best_paths
    log(f"[MAP] Mapped {best_matches} images using column '{best_col}'")
    return df.dropna(subset=["image_path"])

# ------------------------------
# Dataset & Loss
# ------------------------------
class DiamondTier7Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: Dict[str, int], train: bool, img_size: int):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.train = train
        
        # 1. Geometric Transforms (Resize, Flip, Crop)
        if train:
            self.geo_tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ])
        else:
            self.geo_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
            ])

        # Fallback transform if skimage missing
        self.basic_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = self.label2idx[row["tier7"]]
        
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except:
            return self.__getitem__(random.randint(0, len(self.df)-1))

        # Apply Resize BEFORE converting to numpy
        img = self.geo_tf(img)

        # Apply White Balance
        img_np = np.array(img).astype(np.float32) / 255.0
        img_wb = shades_of_gray_wb(img_np) 
        
        # Convert to Lab
        if HAS_SKIMAGE:
            lab = skcolor.rgb2lab(img_wb)
            L = lab[:, :, 0] / 50.0 - 1.0
            a = lab[:, :, 1] / 100.0
            b = lab[:, :, 2] / 100.0
            img_final = np.stack([L, a, b], axis=2).astype(np.float32)
            x = torch.from_numpy(img_final).permute(2, 0, 1)
        else:
            img_pil = Image.fromarray((img_wb * 255).astype(np.uint8))
            x = self.basic_tf(img_pil)

        return x, torch.tensor(y, dtype=torch.long)

class CoralLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.k = num_classes - 1
        
    def forward(self, logits, y):
        b = y.shape[0]
        targets = torch.zeros((b, self.k), device=y.device)
        for i in range(self.k):
            targets[:, i] = (y > i).float()
        return nn.BCEWithLogitsLoss()(logits, targets)

# ------------------------------
# Main Training Loop
# ------------------------------
def main():
    cfg = CFG()
    seed_all(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.out_dir_name)
    out_dir.mkdir(exist_ok=True)
    
    log(f"--- STARTING FINAL TRAINING (Device: {device}) ---")
    
    log("Downloading data...")
    ds_path = Path(kagglehub.dataset_download(cfg.dataset_handle))
    csv_path = list(ds_path.rglob("diamond_data.csv"))[0]
    
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'colour' in df.columns: df.rename(columns={'colour': 'color'}, inplace=True)
    
    df["tier7"] = df["color"].apply(map_color_to_tier7)
    df = df[df["tier7"] != "UNKNOWN"].copy()
    
    root_dir = csv_path.parent
    df = smart_map_images(df, root_dir)
    
    label2idx = {name: i for i, name in enumerate(TIER7_ORDER)}
    df["label_idx"] = df["tier7"].map(label2idx)
    
    # Split
    train_mask = np.random.rand(len(df)) < 0.8
    train_df = df[train_mask].copy()
    val_df = df[~train_mask].copy()
    
    # Weighted Sampler
    class_counts = train_df["label_idx"].value_counts().sort_index()
    log(f"Class Counts: {class_counts.to_dict()}")
    
    weights = 1.0 / class_counts
    sample_weights = train_df["label_idx"].map(weights).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_df),
        replacement=True
    )
    
    train_ds = DiamondTier7Dataset(train_df, label2idx, True, cfg.img_size)
    val_ds = DiamondTier7Dataset(val_df, label2idx, False, cfg.img_size)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # Model
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(TIER7_ORDER)-1)
    model.to(device)
    
    criterion = CoralLoss(len(TIER7_ORDER))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(device)
    
    best_acc = 0.0
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        
        if epoch == 2:
            log("Unfreezing backbone layers...")
            for param in model.parameters(): param.requires_grad = True
        
        # --- PROGRESS BAR IS HERE ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device):
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
            
            # Update bar with current LR and Avg Loss
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = np.mean(losses[-50:]) if len(losses) > 50 else np.mean(losses)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "LR": f"{current_lr:.1e}"})
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).sum(dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        val_acc = correct / total
        log(f"Epoch {epoch} Result: Val Acc={val_acc*100:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_dict = {
                "model_state": model.state_dict(),
                "labels": TIER7_ORDER,
                "config": vars(cfg),
                "acc": best_acc
            }
            torch.save(save_dict, out_dir / "best_model.pth")
            log(f"--> Saved New Best: {best_acc*100:.2f}%")
            
    log(f"DONE. Best Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()