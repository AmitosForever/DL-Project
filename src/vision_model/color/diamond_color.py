import os
import glob
import re
import time
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

import kagglehub

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 0) CONFIG
# ============================================================

@dataclass
class CFG:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    img_size: int = 224

    # Training
    epochs: int = 20
    batch_size: int = 32
    base_lr: float = 2e-4
    weight_decay: float = 0.01
    patience: int = 4
    unfreeze_epoch: int = 2  # unfreeze last blocks after N epochs

    # Sampler
    use_weighted_sampler: bool = True

    # Abstain (optional for final pipeline)
    use_abstain: bool = True
    abstain_threshold: float = 0.55  # if any stage confidence < threshold => abstain

cfg = CFG()

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(cfg.seed)

# ============================================================
# 1) PRICE-BASED 7 TIERS + HIERARCHY
# ============================================================

# 7 price tiers (business meaningful)
TIER7_GROUPS = {
    0: ["D","E","F"],                         # Premium White
    1: ["G","H"],                             # Near Colorless High
    2: ["I","J"],                             # Near Colorless Low
    3: ["K","L"],                             # Faint Yellow
    4: ["M","N"],                             # Very Light Yellow
    5: ["O-P","Q-R"],                         # Light Yellow
    6: ["S-T","U-V","W-X","Y-Z"]              # Yellow Low-end
}
TIER7_NAMES = {
    0: "Premium_White (D/E/F)",
    1: "Near_Colorless_High (G/H)",
    2: "Near_Colorless_Low (I/J)",
    3: "Faint_Yellow (K/L)",
    4: "Very_Light_Yellow (M/N)",
    5: "Light_Yellow (O-P/Q-R)",
    6: "Yellow_LowEnd (S-T..Y-Z)"
}

COLOR_TO_TIER7 = {}
for t, cols in TIER7_GROUPS.items():
    for c in cols:
        COLOR_TO_TIER7[c] = t

# Hierarchical routing:
# A: Premium vs Rest
# B: Near-colorless (tiers 1-2) vs Yellowish (tiers 3-6)
# C_NC: tier1 vs tier2
# C_Y: tiers 3 vs 4 vs 5 vs 6  (4-class)
def label_stage_A(tier7: int) -> int:
    return 1 if tier7 == 0 else 0

def label_stage_B(tier7: int) -> int:
    # only valid if not premium
    return 1 if tier7 in [1,2] else 0  # 1=near-colorless, 0=yellowish

def label_stage_C_NC(tier7: int) -> int:
    # only valid if tier7 in [1,2]
    return 1 if tier7 == 1 else 0  # 1=High(G/H), 0=Low(I/J)

def label_stage_C_Y(tier7: int) -> int:
    # only valid if tier7 in [3,4,5,6]
    # map to 0..3
    mapping = {3:0, 4:1, 5:2, 6:3}
    return mapping[tier7]

C_Y_NAMES = {
    0: "KL (Faint Yellow)",
    1: "MN (Very Light Yellow)",
    2: "OP_QR (Light Yellow)",
    3: "ST_to_YZ (Yellow Low-end)"
}

# ============================================================
# 2) SIMPLE SEGMENTATION + COLOR FEATURES
# ============================================================

def estimate_background_rgb(img_rgb: np.ndarray, border: int = 6) -> np.ndarray:
    h, w, _ = img_rgb.shape
    b = border
    top = img_rgb[:b, :, :]
    bottom = img_rgb[h-b:h, :, :]
    left = img_rgb[:, :b, :]
    right = img_rgb[:, w-b:w, :]
    border_pixels = np.concatenate([top.reshape(-1,3), bottom.reshape(-1,3),
                                    left.reshape(-1,3), right.reshape(-1,3)], axis=0)
    return border_pixels.mean(axis=0)

def simple_diamond_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    A robust-enough mask for this dataset:
    - estimate background from borders
    - mark pixels far from background color as foreground
    - morphological clean-up
    - additionally suppress extreme highlights/shadows using L*
    """
    img = img_rgb.astype(np.float32)
    bg = estimate_background_rgb(img_rgb).astype(np.float32)

    dist = np.sqrt(((img - bg) ** 2).sum(axis=2))
    # threshold: tuned to be conservative
    fg = (dist > 25.0).astype(np.uint8) * 255

    fg = cv2.medianBlur(fg, 5)
    kernel = np.ones((5,5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Lightness gating: remove shiny highlights + deep shadows
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:,:,0]
    gate = ((L > 50) & (L < 235)).astype(np.uint8) * 255

    mask = cv2.bitwise_and(fg, gate)
    # ensure non-empty
    if mask.sum() < 500:
        # fallback: use gate only
        mask = gate
    if mask.sum() < 500:
        # final fallback: everything
        mask = np.ones((img_rgb.shape[0], img_rgb.shape[1]), np.uint8) * 255

    return mask

def extract_color_features(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    32-dim color descriptor based on Lab and HSV within the mask.
    Goal: stable yellowness/body color, not reflections.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    m = (mask > 0)
    if m.sum() < 200:
        m = np.ones_like(m, dtype=bool)

    bvals = b[m]
    Lvals = L[m]

    # b* histogram in focused range around neutral (128)
    # We'll use a wide range to capture more variation.
    hist_bins = 16
    hist = cv2.calcHist([b.astype(np.float32)], [0], mask, [hist_bins], [90, 170]).flatten()
    hist = hist / (hist.sum() + 1e-8)

    # robust stats on b*
    qs = np.percentile(bvals, [5,10,25,50,75,90,95])
    b_mean = float(np.mean(bvals))
    b_std  = float(np.std(bvals))

    # ratio signal (yellowness normalized by lightness)
    ratio = (bvals - 128.0) / (Lvals + 1e-6)
    r_mean = float(np.mean(ratio))
    r_std  = float(np.std(ratio))

    # HSV saturation stats (sometimes helps)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[:,:,1][m]
    V = hsv[:,:,2][m]
    s_mean = float(np.mean(S))
    s_std  = float(np.std(S))
    v_mean = float(np.mean(V))
    v_std  = float(np.std(V))

    feats = np.concatenate([
        hist,                              # 16
        np.array([b_mean, b_std]),         # 2  => 18
        qs.astype(np.float32),             # 7  => 25
        np.array([r_mean, r_std]),         # 2  => 27
        np.array([s_mean, s_std, v_mean, v_std], dtype=np.float32)  # 4 => 31
    ], axis=0)

    # pad to 32
    if feats.shape[0] < 32:
        feats = np.concatenate([feats, np.zeros(32 - feats.shape[0], dtype=np.float32)], axis=0)
    feats = feats[:32].astype(np.float32)
    return feats

def apply_mask_to_rgb(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Keep diamond region, replace background with neutral gray
    """
    bg = np.full_like(img_rgb, 128)
    m = (mask > 0)[:,:,None]
    out = np.where(m, img_rgb, bg)
    return out

# ============================================================
# 3) DATA BUILDING (MATCH CSV -> IMAGES)
# ============================================================

def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

def load_and_match_dataset():
    print("\n=== Downloading dataset ===")
    t0 = time.time()
    path = kagglehub.dataset_download("aayushpurswani/diamond-images-dataset")
    print(f"Download done in {time.time()-t0:.1f}s | Path: {path}")

    csv_path = glob.glob(os.path.join(path, "**", "diamond_data.csv"), recursive=True)[0]
    root_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "colour" in df.columns:
        df.rename(columns={"colour": "color"}, inplace=True)

    # Keep only colors that we can map
    df = df[df["color"].isin(COLOR_TO_TIER7.keys())].copy()
    df["tier7"] = df["color"].map(COLOR_TO_TIER7)

    print("\n=== Raw color distribution (mapped to tier7) ===")
    print(df["color"].value_counts())

    folders = {d.lower(): os.path.join(root_dir, d)
               for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d))}

    matched = []
    print("\n=== Matching images by shape folders ===")
    for shape in df["shape"].unique():
        shape_l = str(shape).lower()
        if shape_l not in folders:
            continue
        folder = folders[shape_l]
        imgs = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))],
                      key=natural_key)
        rows = df[df["shape"].str.lower() == shape_l]

        n = min(len(imgs), len(rows))
        if n <= 0:
            continue

        for i in range(n):
            row = rows.iloc[i].to_dict()
            row["image_path"] = os.path.join(folder, imgs[i])
            matched.append(row)

    mdf = pd.DataFrame(matched)
    print(f"[DATA] Matched samples: {len(mdf)}")

    # add stage labels
    mdf["A_premium"] = mdf["tier7"].apply(label_stage_A)
    # B only for non-premium
    mdf["B_nearcolor"] = mdf["tier7"].apply(lambda t: label_stage_B(t) if t != 0 else -1)
    # C_NC only for tiers 1-2
    mdf["C_nc_high"] = mdf["tier7"].apply(lambda t: label_stage_C_NC(t) if t in [1,2] else -1)
    # C_Y only for tiers 3-6
    mdf["C_y_class"] = mdf["tier7"].apply(lambda t: label_stage_C_Y(t) if t in [3,4,5,6] else -1)

    print("\n=== Tier7 distribution (7 price groups) ===")
    vc = mdf["tier7"].value_counts().sort_index()
    for k in vc.index:
        print(f"{k} | {TIER7_NAMES[k]:28s} : {vc[k]}")
    return mdf

# ============================================================
# 4) PYTORCH DATASETS (STAGE-SPECIFIC)
# ============================================================

class StageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str, img_tf, debug_name=""):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.img_tf = img_tf
        self.debug_name = debug_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except:
            # corrupted -> random replacement
            ridx = random.randint(0, len(self.df)-1)
            return self.__getitem__(ridx)

        mask = simple_diamond_mask(img)
        feats = extract_color_features(img, mask)
        img2 = apply_mask_to_rgb(img, mask)

        img_pil = Image.fromarray(img2)
        x = self.img_tf(img_pil)
        f = torch.tensor(feats, dtype=torch.float32)

        y = int(row[self.label_col])
        return x, f, torch.tensor(y, dtype=torch.long)

def make_transforms(train: bool):
    t = [transforms.Resize((cfg.img_size, cfg.img_size))]
    if train:
        # very light augmentation (avoid killing color)
        t += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ]
    t += [
        transforms.ToTensor(),
        # keep normalization mild (preserve hue ratios)
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]
    return transforms.Compose(t)

# ============================================================
# 5) MODEL: EfficientNet Fusion (image encoder + color MLP)
# ============================================================

class EfficientNetFusion(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        # feature extractor
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.img_dim = 1280  # efficientnet_b0 last feature dim

        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU()
        )

        self.head = nn.Sequential(
            nn.Linear(self.img_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_feat):
        x = self.features(x_img)
        x = self.pool(x).flatten(1)
        f = self.feat_mlp(x_feat)
        z = torch.cat([x, f], dim=1)
        return self.head(z)

def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_last_blocks(model: EfficientNetFusion, n_blocks: int = 2):
    # EfficientNet features is a Sequential of blocks; unfreeze last n blocks
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze head + feat mlp always
    for p in model.feat_mlp.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    # unfreeze last blocks
    blocks = list(model.features.children())
    for b in blocks[-n_blocks:]:
        for p in b.parameters():
            p.requires_grad = True

# ============================================================
# 6) TRAINING UTILITIES
# ============================================================

def make_weighted_sampler(df: pd.DataFrame, label_col: str):
    labels = df[label_col].astype(int).values
    classes, counts = np.unique(labels, return_counts=True)
    count_map = {c: cnt for c, cnt in zip(classes, counts)}
    weights = np.array([1.0 / count_map[y] for y in labels], dtype=np.float64)
    weights = weights / weights.sum()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    print(f"[SAMPLER] {label_col} counts:", dict(zip(classes.tolist(), counts.tolist())))
    return sampler

@torch.no_grad()
def evaluate(model, loader, num_classes: int, device: str, name="VAL"):
    model.eval()
    all_y = []
    all_p = []
    all_prob = []
    for x, f, y in loader:
        x, f = x.to(device), f.to(device)
        logits = model(x, f)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1).cpu().numpy()

        all_y.append(y.numpy())
        all_p.append(pred)
        all_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    probs = np.concatenate(all_prob)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred) if num_classes > 1 else acc
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return acc, bacc, cm, probs, y_true, y_pred

def train_one_stage(stage_name: str,
                    train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    label_col: str,
                    num_classes: int,
                    save_path: str):
    print("\n" + "="*80)
    print(f"STAGE {stage_name} | label={label_col} | classes={num_classes}")
    print("="*80)
    print(f"[DATA] Train: {len(train_df)} | Val: {len(val_df)}")

    img_tf_train = make_transforms(train=True)
    img_tf_val = make_transforms(train=False)

    ds_train = StageDataset(train_df, label_col, img_tf_train, debug_name=stage_name)
    ds_val   = StageDataset(val_df, label_col, img_tf_val, debug_name=stage_name)

    if cfg.use_weighted_sampler:
        sampler = make_weighted_sampler(train_df, label_col)
        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size,
                                  sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = EfficientNetFusion(num_classes=num_classes).to(cfg.device)

    # start: freeze all, train head+feat mlp
    unfreeze_last_blocks(model, n_blocks=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_acc = -1.0
    bad_epochs = 0

    scaler = torch.amp.GradScaler(enabled=(cfg.device=="cuda"))

    for epoch in range(1, cfg.epochs+1):
        t_epoch = time.time()

        # Unfreeze last blocks after some epochs
        if epoch == cfg.unfreeze_epoch:
            print(f"[UNFREEZE] epoch={epoch}: unfreezing last 2 EfficientNet blocks + head")
            unfreeze_last_blocks(model, n_blocks=2)
            optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg.base_lr * 0.5, weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        model.train()
        running = 0.0
        n_batches = 0
        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"{stage_name} Ep {epoch}/{cfg.epochs}", dynamic_ncols=True)
        for x, f, y in pbar:
            x, f, y = x.to(cfg.device), f.to(cfg.device), y.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=cfg.device, enabled=(cfg.device=="cuda")):
                logits = model(x, f)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{running/max(1,n_batches):.4f}",
                "lr": f"{current_lr:.1e}"
            })

        # VAL
        acc, bacc, cm, _, _, _ = evaluate(model, val_loader, num_classes, cfg.device, name="VAL")
        scheduler.step(acc)

        dt = time.time() - t_epoch
        print(f"[{stage_name}] Epoch {epoch:02d} | TrainLoss={running/max(1,n_batches):.4f} | "
              f"ValAcc={acc*100:.2f}% | ValBAcc={bacc*100:.2f}% | LR={optimizer.param_groups[0]['lr']:.1e} | "
              f"Time={dt:.1f}s")

        # print confusion matrix small
        print(f"[{stage_name}] Confusion matrix (rows=true, cols=pred):")
        print(cm)

        if acc > best_acc + 1e-5:
            best_acc = acc
            bad_epochs = 0
            torch.save({"model": model.state_dict()}, save_path)
            print(f"✓ saved best model -> {save_path} | best ValAcc={best_acc*100:.2f}%")
        else:
            bad_epochs += 1
            print(f"[{stage_name}] no improvement | bad_epochs={bad_epochs}/{cfg.patience}")
            if bad_epochs >= cfg.patience:
                print(f"[EARLY STOP] {stage_name} stopped. Best ValAcc={best_acc*100:.2f}%")
                break

    # load best
    ckpt = torch.load(save_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    return model

# ============================================================
# 7) PIPELINE EVALUATION (7-tier prediction + optional abstain)
# ============================================================

@torch.no_grad()
def predict_stage(model, x, f):
    logits = model(x, f)
    prob = torch.softmax(logits, dim=1)
    conf, pred = torch.max(prob, dim=1)
    return pred, conf, prob

@torch.no_grad()
def evaluate_full_pipeline(df_val: pd.DataFrame,
                           model_A,
                           model_B,
                           model_CNC,
                           model_CY):
    """
    Compute final 7-tier accuracy (Top-1) by routing through hierarchy.
    Optionally abstain when confidence < threshold at any stage.
    """
    print("\n" + "="*80)
    print("FINAL PIPELINE EVALUATION (7-tier Top-1 + optional abstain)")
    print("="*80)

    img_tf = make_transforms(train=False)
    ds = StageDataset(df_val, label_col="tier7", img_tf=img_tf, debug_name="PIPELINE")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    y_true_all = []
    y_pred_all = []
    abstained = 0
    total = 0

    for x, f, y_tier7 in tqdm(loader, desc="PIPELINE", dynamic_ncols=True):
        x, f = x.to(cfg.device), f.to(cfg.device)
        y_tier7 = y_tier7.numpy()

        # Stage A
        predA, confA, _ = predict_stage(model_A, x, f)  # 1=premium else 0
        predA = predA.cpu().numpy()
        confA = confA.cpu().numpy()

        # initialize predictions
        out = np.full_like(y_tier7, fill_value=-1)

        for i in range(len(y_tier7)):
            total += 1

            # abstain check
            if cfg.use_abstain and confA[i] < cfg.abstain_threshold:
                abstained += 1
                continue

            if predA[i] == 1:
                out[i] = 0  # Premium tier7
                continue

            # Stage B (only if non-premium)
            xb = x[i:i+1]
            fb = f[i:i+1]
            predB, confB, _ = predict_stage(model_B, xb, fb)  # 1=near-colorless, 0=yellowish
            predB = int(predB.item())
            confB = float(confB.item())

            if cfg.use_abstain and confB < cfg.abstain_threshold:
                abstained += 1
                continue

            if predB == 1:
                # near-colorless -> C_NC
                predC, confC, _ = predict_stage(model_CNC, xb, fb)  # 1=High(G/H)->tier1 else tier2
                predC = int(predC.item())
                confC = float(confC.item())
                if cfg.use_abstain and confC < cfg.abstain_threshold:
                    abstained += 1
                    continue
                out[i] = 1 if predC == 1 else 2
            else:
                # yellowish -> C_Y
                predY, confY, _ = predict_stage(model_CY, xb, fb)  # 0..3 -> tiers 3..6
                predY = int(predY.item())
                confY = float(confY.item())
                if cfg.use_abstain and confY < cfg.abstain_threshold:
                    abstained += 1
                    continue
                out[i] = {0:3, 1:4, 2:5, 3:6}[predY]

        # collect non-abstained
        for i in range(len(y_tier7)):
            if out[i] != -1:
                y_true_all.append(int(y_tier7[i]))
                y_pred_all.append(int(out[i]))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    coverage = (len(y_true_all) / max(1, total)) * 100.0
    acc = accuracy_score(y_true_all, y_pred_all) if len(y_true_all) else 0.0
    bacc = balanced_accuracy_score(y_true_all, y_pred_all) if len(y_true_all) else 0.0

    print(f"\n[PIPELINE] Coverage (not abstained): {coverage:.2f}%")
    print(f"[PIPELINE] 7-tier Top-1 Accuracy (on covered): {acc*100:.2f}%")
    print(f"[PIPELINE] 7-tier Balanced Acc (on covered): {bacc*100:.2f}%")

    print("\n[PIPELINE] Confusion matrix (tiers 0..6):")
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(7)))
    print(cm)

    # per-tier accuracy
    print("\n[PIPELINE] Per-tier accuracy:")
    for t in range(7):
        mask = (y_true_all == t)
        if mask.sum() == 0:
            continue
        a = (y_pred_all[mask] == t).mean() * 100.0
        print(f"{t} | {TIER7_NAMES[t]:28s} : {a:.2f}%  (n={mask.sum()})")

# ============================================================
# 8) MAIN: BUILD DF -> SPLIT -> TRAIN STAGES -> EVAL PIPELINE
# ============================================================

def main():
    mdf = load_and_match_dataset()

    # Stratified split by tier7 (important)
    train_df, val_df = train_test_split(
        mdf, test_size=0.2, random_state=cfg.seed, stratify=mdf["tier7"]
    )
    print(f"\n[SPLIT] Train={len(train_df)} | Val={len(val_df)}")

    # -------------------------
    # Stage A: Premium vs Rest
    # label_col = A_premium (1 premium, 0 rest)
    # -------------------------
    model_A = train_one_stage(
        stage_name="A (Premium vs Rest)",
        train_df=train_df,
        val_df=val_df,
        label_col="A_premium",
        num_classes=2,
        save_path="best_stage_A.pth"
    )

    # -------------------------
    # Stage B: Near-colorless vs Yellowish (only non-premium)
    # -------------------------
    train_B = train_df[train_df["tier7"] != 0].copy()
    val_B   = val_df[val_df["tier7"] != 0].copy()

    model_B = train_one_stage(
        stage_name="B (NearColorless vs Yellowish | on Rest)",
        train_df=train_B,
        val_df=val_B,
        label_col="B_nearcolor",
        num_classes=2,
        save_path="best_stage_B.pth"
    )

    # -------------------------
    # Stage C_NC: tier1 vs tier2 (G/H vs I/J)
    # -------------------------
    train_CNC = train_df[train_df["tier7"].isin([1,2])].copy()
    val_CNC   = val_df[val_df["tier7"].isin([1,2])].copy()

    model_CNC = train_one_stage(
        stage_name="C_NC (G/H vs I/J)",
        train_df=train_CNC,
        val_df=val_CNC,
        label_col="C_nc_high",
        num_classes=2,
        save_path="best_stage_CNC.pth"
    )

    # -------------------------
    # Stage C_Y: tiers 3..6 (KL vs MN vs OP/QR vs ST..YZ)
    # -------------------------
    train_CY = train_df[train_df["tier7"].isin([3,4,5,6])].copy()
    val_CY   = val_df[val_df["tier7"].isin([3,4,5,6])].copy()

    # For visibility: print counts
    print("\n[C_Y] counts in train:")
    print(train_CY["tier7"].value_counts().sort_index().rename(TIER7_NAMES))
    print("[C_Y] counts in val:")
    print(val_CY["tier7"].value_counts().sort_index().rename(TIER7_NAMES))

    # We train C_Y on label_col C_y_class (0..3)
    model_CY = train_one_stage(
        stage_name="C_Y (KL vs MN vs OP/QR vs ST..YZ)",
        train_df=train_CY,
        val_df=val_CY,
        label_col="C_y_class",
        num_classes=4,
        save_path="best_stage_CY.pth"
    )

    # -------------------------
    # Final pipeline evaluation on full VAL
    # -------------------------
    evaluate_full_pipeline(val_df, model_A, model_B, model_CNC, model_CY)

    print("\nDONE.")
    print("Saved checkpoints:")
    print(" - best_stage_A.pth")
    print(" - best_stage_B.pth")
    print(" - best_stage_CNC.pth")
    print(" - best_stage_CY.pth")

if __name__ == "__main__":
    main()
