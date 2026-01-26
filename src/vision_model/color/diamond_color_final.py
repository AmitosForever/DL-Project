import os
import glob
import re
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import kagglehub

try:
    from skimage import color
except ImportError:
    raise ImportError("Please install scikit-image: pip install scikit-image")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F

# =============================================================================
# 1. CONFIG
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

CFG = {
    "dataset_handle": "aayushpurswani/diamond-images-dataset",
    "classes": ["D","E","F","G","H","I","J"],
    "img_size": 224,
    "batch_size": 32,
    "epochs": 50,
    "lr_head": 2e-4,
    "lr_backbone": 1e-5,
    "weight_decay": 0.01,
    "sigma": 0.7,
    "val_split": 0.15,
    "unfreeze_epoch": 1,
    "unfreeze_last_blocks": 2,
    "early_stop_patience": 6,
    "save_path": "best_dinov2_color_top3.pth",
    "seed": 42,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])

# =============================================================================
# 2. GAUSSIAN TARGETS
# =============================================================================
def generate_gaussian_target(true_index, num_classes, sigma):
    x = np.arange(num_classes)
    target = np.exp(-0.5 * ((x - true_index) / sigma) ** 2)
    target = target / (target.sum() + 1e-9)
    return torch.tensor(target, dtype=torch.float32)

# =============================================================================
# 3. TRANSFORMS
# =============================================================================
class ToEnhancedLab(object):
    def __call__(self, img):
        img_np = np.array(img)
        lab = color.rgb2lab(img_np)
        L = lab[:, :, 0] / 100.0
        b = lab[:, :, 2]
        b_norm = (b + 20.0) / 40.0
        b_enh = np.clip(b_norm, 0.0, 1.0)
        out = np.stack([L, b_enh, b_enh], axis=2)
        return torch.from_numpy(out).permute(2, 0, 1).float()

def get_transforms(split="train"):
    t = [transforms.Resize((CFG["img_size"], CFG["img_size"]))]
    if split == "train":
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]
    t += [
        ToEnhancedLab(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]
    return transforms.Compose(t)

# =============================================================================
# 4. DATASET
# =============================================================================
class UnifiedColorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, sigma=0.7):
        self.transform = transform
        self.sigma = sigma

        df = pd.read_csv(csv_file)
        df.columns = [c.lower().strip() for c in df.columns]
        if "colour" in df.columns:
            df.rename(columns={"colour":"color"}, inplace=True)

        cleaning_map = {'D:P:BN':'D','I:P':'I','K:P':'K','J:P':'J'}
        df["color"] = df["color"].replace(cleaning_map)
        df = df[df["color"].isin(CFG["classes"])].copy()

        available_folders = {
            d.lower(): os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        }

        def natural_key(text):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

        matched = []
        for shape in df["shape"].dropna().unique():
            shape_str = str(shape).lower()
            if shape_str not in available_folders:
                continue

            folder = available_folders[shape_str]
            images = sorted(
                [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png",".jpeg"))],
                key=natural_key
            )
            shape_rows = df[df["shape"].str.lower() == shape_str]
            n = min(len(images), len(shape_rows))

            for i in range(n):
                rec = shape_rows.iloc[i].to_dict()
                rec["image_path"] = os.path.join(folder, images[i])
                matched.append(rec)

        self.df = pd.DataFrame(matched)
        if len(self.df) == 0:
            raise RuntimeError("No matched samples found")

        grade_to_idx = {g:i for i,g in enumerate(CFG["classes"])}
        self.df["color_idx"] = self.df["color"].map(grade_to_idx).astype(int)
        self.num_classes = len(CFG["classes"])

        print(f"[DATA] Samples: {len(self.df)} | Classes: {CFG['classes']}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            return self.__getitem__(random.randint(0, len(self.df)-1))

        if self.transform:
            img = self.transform(img)

        true_idx = int(row["color_idx"])
        target = generate_gaussian_target(true_idx, self.num_classes, self.sigma)
        return img, target, torch.tensor(true_idx, dtype=torch.long)

# =============================================================================
# 5. MODEL
# =============================================================================
class DinoV2ColorModel(nn.Module):
    def __init__(self, num_classes, unfreeze_last_blocks=0):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in self.backbone.parameters():
            p.requires_grad = False

        if unfreeze_last_blocks > 0:
            for block in self.backbone.blocks[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        self.head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# =============================================================================
# 6. METRICS
# =============================================================================
@torch.no_grad()
def eval_topk(model, loader, k=3):
    model.eval()
    c1 = ck = total = 0
    for img, _, y in loader:
        img, y = img.to(DEVICE), y.to(DEVICE)
        probs = model(img).exp()
        top1 = probs.argmax(1)
        topk = probs.topk(k, dim=1).indices
        c1 += (top1 == y).sum().item()
        ck += (topk == y.view(-1,1)).any(dim=1).sum().item()
        total += y.size(0)
    return c1/total, ck/total

# =============================================================================
# 7. TRAINING
# =============================================================================
def run_training():
    path = kagglehub.dataset_download(CFG["dataset_handle"])
    csv_path = glob.glob(os.path.join(path,"**","diamond_data.csv"),recursive=True)[0]
    root_dir = os.path.dirname(csv_path)

    full_ds = UnifiedColorDataset(
        csv_file=csv_path,
        root_dir=root_dir,
        transform=get_transforms("train"),
        sigma=CFG["sigma"]
    )

    tr_len = int((1-CFG["val_split"])*len(full_ds))
    va_len = len(full_ds) - tr_len
    tr_ds, va_ds = random_split(full_ds, [tr_len, va_len])
    va_ds.dataset.transform = get_transforms("val")

    tr_loader = DataLoader(tr_ds, CFG["batch_size"], True, num_workers=NUM_WORKERS, pin_memory=True)
    va_loader = DataLoader(va_ds, CFG["batch_size"], False, num_workers=NUM_WORKERS, pin_memory=True)

    model = DinoV2ColorModel(full_ds.num_classes, 0).to(DEVICE)

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": CFG["lr_backbone"]},
            {"params": model.head.parameters(), "lr": CFG["lr_head"]},
        ],
        weight_decay=CFG["weight_decay"]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.amp.GradScaler()

    best_top3 = 0.0
    patience = 0

    for epoch in range(CFG["epochs"]):
        if epoch == CFG["unfreeze_epoch"] and CFG["unfreeze_last_blocks"] > 0:
            for block in model.backbone.blocks[-CFG["unfreeze_last_blocks"]:]:
                for p in block.parameters():
                    p.requires_grad = True
            print(f"Unfroze last {CFG['unfreeze_last_blocks']} blocks")

        model.train()
        loss_sum = 0
        n = 0

        for img, tgt, _ in tqdm(tr_loader, desc=f"Ep {epoch+1}/{CFG['epochs']}"):
            img, tgt = img.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if DEVICE=="cuda" else "cpu"):
                logp = model(img)
                loss = criterion(logp, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            n += 1

        acc1, acc3 = eval_topk(model, va_loader, 3)
        print(f"Train KL {loss_sum/n:.4f} | Val Top-1 {acc1:.2%} | Val Top-3 {acc3:.2%}")
        scheduler.step(acc3)

        if acc3 > best_top3:
            best_top3 = acc3
            patience = 0
            torch.save(model.state_dict(), CFG["save_path"])
            print(f"✓ Saved best model ({best_top3:.2%})")
        else:
            patience += 1

        if patience >= CFG["early_stop_patience"]:
            print("Early stopping")
            break

    print(f"BEST TOP-3: {best_top3:.2%}")

# =============================================================================
if __name__ == "__main__":
    run_training()
