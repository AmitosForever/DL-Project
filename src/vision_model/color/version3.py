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
    raise ImportError("pip install scikit-image")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import torch.nn.functional as F

# =============================================================================
# 1. CONFIG
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

FULL_COLOR_CLASSES = [
    "D","E","F","G","H","I","J","K","L","M","N",
    "O-P","Q-R","S-T","U-V","W-X","Y-Z"
]

CFG = {
    "dataset_handle": "aayushpurswani/diamond-images-dataset",
    "classes": FULL_COLOR_CLASSES,
    "img_size": 224,
    "batch_size": 32,
    "epochs": 50,
    "lr_head": 2e-4,
    "lr_backbone": 1e-5,
    "weight_decay": 0.01,
    "sigma": 1.3,
    "val_split": 0.15,
    "unfreeze_epoch": 1,
    "unfreeze_last_blocks": 2,
    "early_stop_patience": 7,
    "save_path": "best_dinov2_color_full_top3.pth",
    "seed": 42,
}

# =============================================================================
# 2. UTILS
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])

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
        b_norm = (b + 40.0) / 80.0
        b_enh = np.clip(b_norm, 0.0, 1.0)
        out = np.stack([L, b_enh, b_enh], axis=2)
        return torch.from_numpy(out).permute(2, 0, 1).float()

def get_transforms(split="train"):
    t = [transforms.Resize((CFG["img_size"], CFG["img_size"]))]
    if split == "train":
        t += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
    t += [ToEnhancedLab(), transforms.Normalize([0.5]*3, [0.5]*3)]
    return transforms.Compose(t)

# =============================================================================
# 4. DATASET
# =============================================================================
def normalize_color_label(x):
    if not isinstance(x, str): return ""
    s = x.strip().upper().replace("–", "-").replace(" ", "")
    cleaning = {
        "D:P:BN":"D","I:P":"I","J:P":"J","K:P":"K",
        "O-P:":"O-P","Q-R:":"Q-R","S-T:":"S-T",
        "U-V:":"U-V","W-X:":"W-X","Y-Z:":"Y-Z"
    }
    return cleaning.get(s, s)

class UnifiedColorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, sigma=1.3):
        self.transform = transform
        self.sigma = sigma
        self.classes = CFG["classes"]
        self.grade_to_idx = {g:i for i,g in enumerate(self.classes)}

        df = pd.read_csv(csv_file)
        df.columns = [c.lower().strip() for c in df.columns]
        if "colour" in df.columns:
            df.rename(columns={"colour":"color"}, inplace=True)

        df["color"] = df["color"].apply(normalize_color_label)
        df = df[df["color"].isin(self.classes)].copy()

        folders = {
            d.lower(): os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        }

        def natural_key(t): return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", t)]

        rows = []
        for shape in df["shape"].dropna().unique():
            s = str(shape).lower()
            if s not in folders: continue
            imgs = sorted(
                [f for f in os.listdir(folders[s]) if f.lower().endswith((".jpg",".png",".jpeg"))],
                key=natural_key
            )
            shape_rows = df[df["shape"].str.lower()==s]
            for i in range(min(len(imgs), len(shape_rows))):
                r = shape_rows.iloc[i].to_dict()
                r["image_path"] = os.path.join(folders[s], imgs[i])
                rows.append(r)

        self.df = pd.DataFrame(rows)
        self.df["color_idx"] = self.df["color"].map(self.grade_to_idx).astype(int)
        self.num_classes = len(self.classes)

        print(f"[DATA] Samples: {len(self.df)} | Classes: {self.classes}")
        print(self.df["color"].value_counts().head(10))

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            return self.__getitem__(random.randint(0,len(self.df)-1))

        if self.transform:
            img = self.transform(img)

        y = int(row["color_idx"])
        tgt = generate_gaussian_target(y, self.num_classes, self.sigma)
        return img, tgt, torch.tensor(y)

# =============================================================================
# 5. MODEL
# =============================================================================
class DinoV2ColorModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in self.backbone.parameters(): p.requires_grad = False
        self.head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(384,256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256,num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x): return self.head(self.backbone(x))

# =============================================================================
# 6. TRAINING
# =============================================================================
def run_training():
    path = kagglehub.dataset_download(CFG["dataset_handle"])
    csv_path = glob.glob(os.path.join(path,"**","diamond_data.csv"),recursive=True)[0]
    root_dir = os.path.dirname(csv_path)

    full_ds = UnifiedColorDataset(csv_path, root_dir, get_transforms("train"), CFG["sigma"])
    tr_len = int((1-CFG["val_split"])*len(full_ds))
    va_len = len(full_ds)-tr_len
    tr_ds, va_ds = random_split(full_ds,[tr_len,va_len])
    va_ds.dataset.transform = get_transforms("val")

    # -------- WeightedRandomSampler --------
    train_labels = [full_ds.df.iloc[i]["color_idx"] for i in tr_ds.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / np.maximum(class_counts,1)
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    tr_loader = DataLoader(tr_ds, CFG["batch_size"], sampler=sampler,
                           num_workers=NUM_WORKERS, pin_memory=True)
    va_loader = DataLoader(va_ds, CFG["batch_size"], shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    model = DinoV2ColorModel(full_ds.num_classes).to(DEVICE)

    optimizer = optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": CFG["lr_backbone"]},
         {"params": model.head.parameters(), "lr": CFG["lr_head"]}],
        weight_decay=CFG["weight_decay"]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))

    best_top3 = 0.0
    patience = 0

    for epoch in range(CFG["epochs"]):
        model.train()
        loss_sum = n = 0
        current_lr = optimizer.param_groups[1]["lr"]

        pbar = tqdm(tr_loader, desc=f"Ep {epoch+1}/{CFG['epochs']}")
        for img, tgt, _ in pbar:
            img, tgt = img.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
                loss = criterion(model(img), tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item(); n+=1
            pbar.set_postfix({"kl_loss":f"{loss_sum/n:.4f}", "lr":f"{current_lr:.1e}"})

        # ---- validation ----
        model.eval()
        tot=top3=0
        with torch.no_grad():
            for img,_,y in va_loader:
                img,y=img.to(DEVICE),y.to(DEVICE)
                probs=model(img).exp()
                top3+=(probs.topk(3,1).indices==y.view(-1,1)).any(1).sum().item()
                tot+=y.size(0)
        acc3=top3/tot
        print(f"Val Top-3: {acc3:.2%}")

        scheduler.step(acc3)
        if acc3>best_top3:
            best_top3=acc3; patience=0
            torch.save(model.state_dict(), CFG["save_path"])
            print("✓ saved best model")
        else:
            patience+=1
        if patience>=CFG["early_stop_patience"]:
            print("Early stopping"); break

    print(f"BEST TOP-3: {best_top3:.2%}")

# =============================================================================
if __name__ == "__main__":
    run_training()
