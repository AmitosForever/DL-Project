# diamond_color_realmap_train.py
# ------------------------------------------------------------
# Goal:
# 1) Try to discover REAL mapping between CSV rows and image files.
# 2) If mapping is found (high match ratio), train a CNN for tier7.
# 3) If not found -> abort with clear diagnostics.
# 4) Save outputs next to THIS script (not the current working directory).
# ------------------------------------------------------------

import os
import re
import time
import json
import math
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

try:
    import kagglehub
except Exception as e:
    raise ImportError("Need kagglehub: pip install kagglehub") from e

# Optional, for better color spaces. If missing, we fallback to RGB.
try:
    from skimage import color as skcolor
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


# ------------------------------
# Pretty logging
# ------------------------------
def log(msg: str):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_script_dir() -> Path:
    # Works for normal python execution. Fallback to cwd if __file__ doesn't exist (rare).
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


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
    num_workers: int = 2

    lr: float = 2e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.03

    # If mapping success ratio < this -> abort (labels not tied to images)
    min_match_ratio: float = 0.80

    # Weighted sampler
    use_weighted_sampler: bool = True

    # Train tricks
    unfreeze_at: int = 2  # unfreeze backbone after N epochs
    early_stop_patience: int = 4

    # Output (will be set to: script_dir / "_realmap_out" by default)
    out_dir: str = ""


cfg = CFG()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------
# Tier mapping (7 price groups)
# ------------------------------
TIER7_ORDER = [
    "Premium_White",         # D/E/F
    "Near_Colorless_High",   # G/H
    "Near_Colorless_Low",    # I/J
    "Faint_Yellow",          # K/L
    "Very_Light_Yellow",     # M/N
    "Light_Yellow",          # O-P / Q-R
    "Yellow_LowEnd",         # S-T .. Y-Z
]


def map_color_to_tier7(color_str: str) -> str:
    c = str(color_str).strip().upper()
    if c in ["D", "E", "F"]:
        return "Premium_White"
    if c in ["G", "H"]:
        return "Near_Colorless_High"
    if c in ["I", "J"]:
        return "Near_Colorless_Low"
    if c in ["K", "L"]:
        return "Faint_Yellow"
    if c in ["M", "N"]:
        return "Very_Light_Yellow"
    if c in ["O-P", "Q-R"]:
        return "Light_Yellow"
    if c in ["S-T", "U-V", "W-X", "Y-Z"]:
        return "Yellow_LowEnd"
    return "UNKNOWN"


# ------------------------------
# Utility: list images
# ------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_all_images(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def normalize_token(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\\", "/")
    return s


def basename_no_ext(p: str) -> str:
    p = normalize_token(p)
    name = p.split("/")[-1]
    name = os.path.splitext(name)[0]
    return name


def find_numeric_tokens(s: str) -> List[str]:
    return re.findall(r"\d+", str(s))


# ------------------------------
# Attempt to discover mapping column
# ------------------------------
def build_image_indexes(image_paths: List[Path]) -> Dict[str, Dict]:
    """
    Build multiple lookup keys -> image path.
    Keys include:
    - full basename without extension
    - basename lower
    - numeric id token if filename contains digits (only if unique)
    """
    by_base: Dict[str, Path] = {}
    by_base_lower: Dict[str, Path] = {}

    by_num: Dict[str, List[Path]] = {}

    for p in image_paths:
        b = p.stem
        by_base[b] = p
        by_base_lower[b.lower()] = p
        nums = find_numeric_tokens(b)
        for t in nums:
            by_num.setdefault(t, []).append(p)

    by_num_unique: Dict[str, Path] = {}
    for t, lst in by_num.items():
        if len(lst) == 1:
            by_num_unique[t] = lst[0]

    return {
        "__by_base__": by_base,
        "__by_base_lower__": by_base_lower,
        "__by_num_unique__": by_num_unique,
    }


def score_mapping_for_column(df: pd.DataFrame, col: str, img_index: Dict[str, Dict], root: Path) -> Tuple[float, str]:
    """
    Try several strategies to map df[col] -> an image path.
    Return (match_ratio, strategy_name).
    """
    series = df[col].astype(str).fillna("").map(normalize_token)

    by_base = img_index["__by_base__"]
    by_base_lower = img_index["__by_base_lower__"]
    by_num_unique = img_index["__by_num_unique__"]

    n = len(df)
    if n == 0:
        return 0.0, "empty_df"

    sample_n = min(2000, n)

    # Strategy A: if it already looks like a path relative to root or absolute
    matched = 0
    for v in series.head(sample_n):
        vp = Path(v)
        if vp.is_file() and vp.suffix.lower() in IMG_EXTS:
            matched += 1
            continue
        rp = (root / v)
        if rp.is_file() and rp.suffix.lower() in IMG_EXTS:
            matched += 1
    ratio_a = matched / sample_n
    best_ratio = ratio_a
    best_name = "path_or_relpath"

    # Strategy B: basename match
    matched = 0
    for v in series.head(sample_n):
        b = basename_no_ext(v)
        if b in by_base:
            matched += 1
    ratio_b = matched / sample_n
    if ratio_b > best_ratio:
        best_ratio = ratio_b
        best_name = "basename"

    # Strategy C: lowercase basename match
    matched = 0
    for v in series.head(sample_n):
        b = basename_no_ext(v).lower()
        if b in by_base_lower:
            matched += 1
    ratio_c = matched / sample_n
    if ratio_c > best_ratio:
        best_ratio = ratio_c
        best_name = "basename_lower"

    # Strategy D: numeric token match (unique only)
    matched = 0
    for v in series.head(sample_n):
        nums = find_numeric_tokens(v)
        ok = any((t in by_num_unique) for t in nums)
        matched += int(ok)
    ratio_d = matched / sample_n
    if ratio_d > best_ratio:
        best_ratio = ratio_d
        best_name = "numeric_token_unique"

    return float(best_ratio), best_name


def materialize_mapping(df: pd.DataFrame, col: str, strategy: str, img_index: Dict[str, Dict], root: Path) -> pd.Series:
    series = df[col].astype(str).fillna("").map(normalize_token)
    by_base = img_index["__by_base__"]
    by_base_lower = img_index["__by_base_lower__"]
    by_num_unique = img_index["__by_num_unique__"]

    out = []
    bad = 0
    for v in series:
        p: Optional[Path] = None

        if strategy == "path_or_relpath":
            vp = Path(v)
            if vp.is_file() and vp.suffix.lower() in IMG_EXTS:
                p = vp
            else:
                rp = (root / v)
                if rp.is_file() and rp.suffix.lower() in IMG_EXTS:
                    p = rp

        elif strategy == "basename":
            b = basename_no_ext(v)
            p = by_base.get(b)

        elif strategy == "basename_lower":
            b = basename_no_ext(v).lower()
            p = by_base_lower.get(b)

        elif strategy == "numeric_token_unique":
            nums = find_numeric_tokens(v)
            for t in nums:
                if t in by_num_unique:
                    p = by_num_unique[t]
                    break

        if p is None:
            bad += 1
            out.append(None)
        else:
            out.append(str(p))

    log(f"[MAP] Materialized mapping using col='{col}' strategy='{strategy}' | missing={bad}/{len(out)}")
    return pd.Series(out, index=df.index)


# ------------------------------
# White balance (simple Shades-of-Gray)
# ------------------------------
def shades_of_gray_wb(img_rgb: np.ndarray, p: int = 6, eps: float = 1e-6) -> np.ndarray:
    """
    img_rgb: HxWx3 float32 in [0,1]
    """
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3
    img = np.clip(img_rgb, 0.0, 1.0)
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p) + eps
    illum = illum / (np.mean(illum) + eps)
    out = img / illum[None, None, :]
    return np.clip(out, 0.0, 1.0)


# ------------------------------
# Dataset
# ------------------------------
class DiamondTier7Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: Dict[str, int], train: bool):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.train = train

        # Keep color info: avoid heavy color jitter
        t = [transforms.Resize((cfg.img_size, cfg.img_size))]
        if train:
            t.append(transforms.RandomHorizontalFlip(p=0.5))
            t.append(transforms.RandomRotation(10))
        t.append(transforms.ToTensor())
        self.base_tf = transforms.Compose(t)

        self.norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["image_path"]
        y_name = row["tier7"]
        y = self.label2idx[y_name]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            ridx = random.randint(0, len(self.df) - 1)
            return self.__getitem__(ridx)

        x = self.base_tf(img)  # tensor [0,1]

        # White balance (numpy)
        x_np = x.permute(1, 2, 0).numpy().astype(np.float32)
        x_np = shades_of_gray_wb(x_np, p=6)
        x = torch.from_numpy(x_np).permute(2, 0, 1)

        # Optional Lab emphasis
        if HAS_SKIMAGE:
            lab = skcolor.rgb2lab(x.permute(1, 2, 0).numpy())
            L = lab[:, :, 0] / 100.0
            a = (lab[:, :, 1] + 128.0) / 255.0
            b = (lab[:, :, 2] + 128.0) / 255.0
            x = torch.tensor(np.stack([L, b, b], axis=2), dtype=torch.float32).permute(2, 0, 1)

        x = self.norm(x)
        return x, torch.tensor(y, dtype=torch.long)


# ------------------------------
# Metrics + Reporting
# ------------------------------
@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, n_classes: int) -> Dict[str, object]:
    model.eval()
    total = 0
    correct = 0

    per_cls_total = np.zeros(n_classes, dtype=np.int64)
    per_cls_correct = np.zeros(n_classes, dtype=np.int64)
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

        y_cpu = y.detach().cpu().numpy()
        p_cpu = pred.detach().cpu().numpy()

        for t, p in zip(y_cpu, p_cpu):
            conf[t, p] += 1

        for c in range(n_classes):
            mask = (y == c)
            ct = mask.sum().item()
            if ct > 0:
                per_cls_total[c] += ct
                per_cls_correct[c] += (pred[mask] == c).sum().item()

    acc = correct / max(1, total)
    per_cls_acc = (per_cls_correct / np.maximum(1, per_cls_total)).astype(float)
    bacc = float(np.mean(per_cls_acc))

    return {
        "acc": float(acc),
        "bacc": float(bacc),
        "per_class_acc": per_cls_acc.tolist(),
        "confusion_matrix": conf.tolist(),
    }


def save_eval_report(out_dir: Path, labels: List[str], metrics: Dict[str, object], prefix: str = "eval") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "acc": metrics["acc"],
        "bacc": metrics["bacc"],
        "labels": labels,
        "per_class_acc": metrics["per_class_acc"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    (out_dir / f"{prefix}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Also dump confusion matrix as CSV for quick viewing
    conf = np.array(metrics["confusion_matrix"], dtype=np.int64)
    df_conf = pd.DataFrame(conf, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    df_conf.to_csv(out_dir / f"{prefix}_confusion.csv", index=True)

    # And per-class acc as CSV
    df_acc = pd.DataFrame({"label": labels, "per_class_acc": metrics["per_class_acc"]})
    df_acc.to_csv(out_dir / f"{prefix}_per_class_acc.csv", index=False)


def make_weighted_sampler(y_train: np.ndarray, n_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    weights_per_class = 1.0 / np.maximum(1.0, counts)
    sample_weights = weights_per_class[y_train]
    log(f"[SAMPLER] class counts={counts.astype(int).tolist()}")
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="store_true", help="Run evaluation only (no training).")
    p.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint .pth")
    p.add_argument("--out_dir", type=str, default=None, help="Override output directory")
    return p.parse_args()


# ------------------------------
# Main pipeline
# ------------------------------
def main(args):
    seed_all(cfg.seed)

    # Performance knobs that don't change correctness
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    script_dir = get_script_dir()
    default_out = script_dir / "_realmap_out"

    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("DIAMOND tier7 TRAIN (REAL-MAP OR ABORT)")
    log(f"DEVICE={DEVICE} | skimage={HAS_SKIMAGE}")
    log(f"OUT_DIR={out_dir}")
    log("=" * 70)

    # 1) Download + locate CSV
    log("Downloading dataset via kagglehub...")
    t0 = time.time()
    ds_path = Path(kagglehub.dataset_download(cfg.dataset_handle))
    log(f"Download done in {time.time() - t0:.1f}s | Path: {ds_path}")

    csv_candidates = list(ds_path.rglob("diamond_data.csv"))
    if not csv_candidates:
        raise FileNotFoundError("Could not find diamond_data.csv under downloaded dataset.")
    csv_path = csv_candidates[0]
    root_dir = csv_path.parent
    log(f"CSV: {csv_path}")
    log(f"ROOT: {root_dir}")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "colour" in df.columns and "color" not in df.columns:
        df.rename(columns={"colour": "color"}, inplace=True)

    if "color" not in df.columns:
        raise ValueError(f"No 'color' column found. columns={df.columns.tolist()}")

    cleaning_map = {"D:P:BN": "D", "I:P": "I", "K:P": "K", "J:P": "J"}
    df["color"] = df["color"].replace(cleaning_map)
    df["tier7"] = df["color"].apply(map_color_to_tier7)
    df = df[df["tier7"] != "UNKNOWN"].copy()

    log("Raw tier7 distribution (CSV):")
    print(df["tier7"].value_counts(), flush=True)

    # 2) List images + build indexes
    log("Scanning image files...")
    imgs = list_all_images(root_dir)
    log(f"Found images: {len(imgs)}")

    if len(imgs) < 1000:
        log("[WARN] Too few images found. Dataset structure may be different than expected.")

    img_index = build_image_indexes(imgs)

    # 3) Discover best mapping column
    candidates = []
    for c in df.columns:
        if any(k in c for k in ["image", "img", "file", "path", "name", "id", "uuid"]):
            candidates.append(c)
    if not candidates:
        candidates = list(df.columns)

    log(f"Mapping candidates (ordered): {candidates[:25]}{' ...' if len(candidates) > 25 else ''}")

    best = (0.0, None, None)  # ratio, col, strategy
    for c in candidates:
        ratio, strat = score_mapping_for_column(df, c, img_index, root_dir)
        log(f"[MAP-SCAN] col='{c}' -> ratio≈{ratio:.3f} via {strat}")
        if ratio > best[0]:
            best = (ratio, c, strat)

    best_ratio, best_col, best_strat = best
    log(f"[MAP-BEST] ratio≈{best_ratio:.3f} col='{best_col}' strat='{best_strat}'")

    if best_col is None or best_ratio < cfg.min_match_ratio:
        log("")
        log("!!! ABORTING !!!")
        log("Could not find a reliable CSV->image mapping column.")
        log("This means labels are likely not tied to images in a supervised way.")
        diag = {
            "best_ratio": best_ratio,
            "best_col": best_col,
            "best_strategy": best_strat,
            "csv_columns": df.columns.tolist(),
            "num_images_found": len(imgs),
            "min_match_ratio_required": cfg.min_match_ratio,
        }
        (out_dir / "mapping_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        log(f"Saved diagnostics -> {out_dir / 'mapping_diagnostics.json'}")
        return

    # 4) Materialize mapping
    df["image_path"] = materialize_mapping(df, best_col, best_strat, img_index, root_dir)
    before = len(df)
    df = df[df["image_path"].notna()].copy()
    df["image_path"] = df["image_path"].astype(str)
    after = len(df)
    log(f"[MAP] Kept mapped rows: {after}/{before} = {after / max(1, before):.2%}")

    # 5) Train/Val split (stratified)
    labels = TIER7_ORDER
    label2idx = {name: i for i, name in enumerate(labels)}
    df["y"] = df["tier7"].map(label2idx).astype(int)

    val_frac = 0.2
    idxs = np.arange(len(df))
    y_all = df["y"].values

    train_idx, val_idx = [], []
    for c in range(len(labels)):
        c_idx = idxs[y_all == c]
        np.random.shuffle(c_idx)
        k = int(len(c_idx) * val_frac)
        val_idx.extend(c_idx[:k].tolist())
        train_idx.extend(c_idx[k:].tolist())

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    log(f"[SPLIT] Train={len(train_df)} | Val={len(val_df)} | Classes={len(labels)}")

    log("[TRAIN] tier7 distribution (train):")
    print(train_df["tier7"].value_counts(), flush=True)
    log("[VAL] tier7 distribution (val):")
    print(val_df["tier7"].value_counts(), flush=True)

    # 6) DataLoaders
    train_ds = DiamondTier7Dataset(train_df, label2idx, train=True)
    val_ds = DiamondTier7Dataset(val_df, label2idx, train=False)

    if cfg.use_weighted_sampler:
        sampler = make_weighted_sampler(train_df["y"].values, n_classes=len(labels))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 7) Model
    log("Loading EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, len(labels))
    )
    model.to(DEVICE)

    # AMP setup (new API if available, fallback otherwise)
    use_amp = (DEVICE == "cuda")
    try:
        autocast_ctx = torch.amp.autocast  # type: ignore[attr-defined]
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]
        amp_mode = "torch.amp"
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        amp_mode = "torch.cuda.amp"
    log(f"[AMP] mode={amp_mode} enabled={use_amp}")

    # If eval: load ckpt and evaluate ONLY
    if args.eval:
        if not args.ckpt:
            raise ValueError("Eval mode requires --ckpt <path_to_pth>")
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        log("=" * 70)
        log(f"[EVAL] Loading checkpoint: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        metrics = eval_model(model, val_loader, n_classes=len(labels))
        log(f"[EVAL] acc={metrics['acc']*100:.2f}% | bacc={metrics['bacc']*100:.2f}%")
        save_eval_report(out_dir, labels, metrics, prefix="eval")
        log(f"[EVAL] Saved report -> {out_dir / 'eval_report.json'}")
        log("=" * 70)
        return

    # Freeze backbone initially (train head first)
    for p in model.features.parameters():
        p.requires_grad = False

    # 8) Loss + optimizer + scheduler
    counts = train_df["y"].value_counts().sort_index()
    freq = np.array([counts.get(i, 0) for i in range(len(labels))], dtype=np.float64)
    cls_w = 1.0 / np.maximum(1.0, freq)
    cls_w = cls_w / cls_w.mean()
    class_weights = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
    log(f"[LOSS] class_weights (normalized): {cls_w.round(3).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_bacc = -1.0
    bad_epochs = 0

    # Always save in the script folder out_dir
    best_path = out_dir / "best_effnet_tier7.pth"

    log("=" * 70)
    log("START TRAIN")
    log("=" * 70)

    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.time()

        if epoch == cfg.unfreeze_at:
            log(f"[UNFREEZE] epoch={epoch}: unfreezing backbone")
            for p in model.features.parameters():
                p.requires_grad = True

        model.train()
        running_loss = 0.0
        n_batches = 0

        current_lr = optimizer.param_groups[0]["lr"]
        log(f"[EPOCH {epoch:02d}/{cfg.epochs}] LR={current_lr:.2e}")

        for bi, (x, y) in enumerate(train_loader, start=1):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(device_type="cuda", enabled=use_amp) if amp_mode == "torch.amp" else autocast_ctx(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

            if bi % 50 == 0 or bi == 1:
                avg_loss = running_loss / max(1, n_batches)
                log(f"  [train] batch {bi}/{len(train_loader)} | loss={avg_loss:.4f}")

        train_loss = running_loss / max(1, n_batches)

        metrics = eval_model(model, val_loader, n_classes=len(labels))
        val_acc = metrics["acc"]
        val_bacc = metrics["bacc"]

        log(f"[VAL] acc={val_acc*100:.2f}% | bacc={val_bacc*100:.2f}% | train_loss={train_loss:.4f} | epoch_time={time.time()-t_ep:.1f}s")

        scheduler.step(val_bacc)

        if val_bacc > best_bacc + 1e-4:
            best_bacc = val_bacc
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            log(f"✓ saved best -> {best_path} | best_bacc={best_bacc*100:.2f}%")

            # Save eval report for the best checkpoint too (guaranteed useful)
            save_eval_report(out_dir, labels, metrics, prefix="best")
            log(f"✓ saved best eval report -> {out_dir / 'best_report.json'}")
        else:
            bad_epochs += 1
            log(f"[NO IMPROVE] bad_epochs={bad_epochs}/{cfg.early_stop_patience}")
            if bad_epochs >= cfg.early_stop_patience:
                log("[EARLY STOP] No improvement.")
                break

    log("=" * 70)
    log("DONE.")
    log(f"Best balanced accuracy: {best_bacc*100:.2f}%")
    log(f"Best checkpoint: {best_path}")
    log("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    main(args)
