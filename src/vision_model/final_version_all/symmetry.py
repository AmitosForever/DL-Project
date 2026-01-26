#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
version1.py  (Image-Only)
Diamond Symmetry Classification (4-class) from IMAGE ONLY.

Dataset: Kaggle "diamond-images-dataset" (aayushpurswani)
Target: symmetry in {Excellent, Very Good, Good, Fair}

Features:
- Runs with ZERO args by default (auto kagglehub download)
- Smart label normalization: Ex / VG / Very Good / etc -> 0..3
- Auto-detect target column containing 'symmetry'
- Auto-detect image path column (e.g., path_to_img)
- WeightedRandomSampler for class imbalance
- Pretrained ConvNeXt-Tiny (torchvision)
- Higher resolution training (default img_size=384)
- Robust image loading: missing/broken/truncated images -> black placeholder (no crash)
- Progress bars (tqdm) for train and validation
- AMP compatible across PyTorch versions (new API if available, else fallback)

Run:
  python version1.py

Optional:
  python version1.py --img_size 512 --batch_size 8
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import os
import random
import re
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from PIL import UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision
from torchvision import transforms

from tqdm import tqdm

# Allow PIL to load truncated JPEGs instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Repro
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# AMP compatibility helpers
# -----------------------------
def make_grad_scaler(device: torch.device):
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")  # newer
    except Exception:
        pass
    try:
        return torch.cuda.amp.GradScaler()  # older
    except Exception:
        return None


def autocast_context(device: torch.device):
    if device.type != "cuda":
        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        return DummyCtx()

    try:
        return torch.amp.autocast("cuda")  # newer
    except Exception:
        pass

    try:
        return torch.cuda.amp.autocast()   # older
    except Exception:
        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        return DummyCtx()


# -----------------------------
# Helpers
# -----------------------------
def is_image_like(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip().lower()
    s = s.split("?")[0]
    return any(s.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"])


def normalize_symmetry_label(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().lower()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = s.replace(".", "")
    s = s.replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if s in {"ex", "exc", "excel", "excellent"} or s.startswith("ex"):
        return "excellent"
    if s in {"vg", "v g", "verygood", "very good"} or ("very" in s and "good" in s) or s.startswith("vg"):
        return "very good"
    if s in {"g", "good"} or s.startswith("good") or s.endswith(" good"):
        return "good"
    if s in {"f", "fair"} or "fair" in s:
        return "fair"
    return None


SYM_STR_TO_ID: Dict[str, int] = {
    "excellent": 0,
    "very good": 1,
    "good": 2,
    "fair": 3,
}
ID_TO_METRIC = {0: "excellent", 1: "very_good", 2: "good", 3: "fair"}


def autodetect_target_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in cols:
        if c.strip().lower() == "symmetry":
            return c
    for c in cols:
        if "symmetry" in c.strip().lower():
            return c
    raise ValueError("Could not auto-detect target column containing 'symmetry'. Use --target_col.")


def autodetect_image_col(df: pd.DataFrame) -> str:
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        raise ValueError("No object/string columns to detect image paths. Use --image_col.")

    ranked = []
    n = len(df)
    sample_n = min(300, n)
    sample_idx = np.linspace(0, n - 1, num=sample_n, dtype=int) if n > 1 else np.array([0], dtype=int)

    for c in obj_cols:
        cl = c.strip().lower()
        name_score = 0
        if "path" in cl:
            name_score += 3
        if "img" in cl or "image" in cl:
            name_score += 2
        if "file" in cl or "filename" in cl:
            name_score += 1

        vals = df[c].iloc[sample_idx].astype(str).fillna("").tolist()
        hits = 0
        for v in vals:
            vv = v.strip()
            if is_image_like(vv):
                hits += 1
            else:
                vv2 = vv.split("?")[0].lower()
                if is_image_like(vv2):
                    hits += 1
        hit_rate = hits / max(1, len(vals))
        ranked.append((name_score + 10 * hit_rate, hit_rate, c))

    ranked.sort(reverse=True)
    _, best_hit, best_col = ranked[0]
    if best_hit < 0.10:
        raise ValueError("Could not reliably auto-detect an image column. Use --image_col.")
    return best_col


def stratified_split_indices(y: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    train_idx = []
    val_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_frac))
        n_val = max(1, n_val) if len(cls_idx) >= 2 else 0
        v = cls_idx[:n_val]
        t = cls_idx[n_val:]
        val_idx.append(v)
        train_idx.append(t)
    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# -----------------------------
# Normalization (robust across torchvision versions)
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_norm_from_weights(weights) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    mean, std = None, None

    try:
        if hasattr(weights, "meta") and isinstance(weights.meta, dict):
            mean = weights.meta.get("mean", None)
            std = weights.meta.get("std", None)
    except Exception:
        mean, std = None, None

    if mean is None or std is None:
        try:
            t = weights.transforms()
            for tr in getattr(t, "transforms", []):
                if isinstance(tr, transforms.Normalize):
                    mean = tuple(float(x) for x in tr.mean)
                    std = tuple(float(x) for x in tr.std)
                    break
        except Exception:
            pass

    if mean is None or std is None:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    return tuple(mean), tuple(std)


# -----------------------------
# kagglehub support
# -----------------------------
def maybe_download_with_kagglehub(dataset: str) -> str:
    try:
        import kagglehub  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "kagglehub לא מותקן. תתקין: pip install kagglehub\n"
            "או שתספק --csv_path ו--img_root ידנית."
        ) from e
    return kagglehub.dataset_download(dataset)


def auto_pick_csv_and_img_root(download_root: str) -> Tuple[str, str]:
    csvs = glob.glob(os.path.join(download_root, "**", "*.csv"), recursive=True)
    if not csvs:
        raise RuntimeError(f"No CSV found under: {download_root}")

    best = None
    best_score = -1
    for p in csvs:
        try:
            df = pd.read_csv(p, nrows=500)
        except Exception:
            continue

        score = 0
        try:
            _ = autodetect_target_col(df)
            score += 3
        except Exception:
            pass
        try:
            _ = autodetect_image_col(df)
            score += 3
        except Exception:
            pass

        if score > best_score:
            best_score = score
            best = p

    if best is None or best_score < 5:
        raise RuntimeError("Could not confidently pick a CSV. Provide --csv_path and --img_root explicitly.")
    return best, download_root


# -----------------------------
# Dataset (image only) with robust image loading
# -----------------------------
class DiamondSymmetryImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        img_root: str,
        image_col: str,
        transform: transforms.Compose,
        fallback_size: int,
        csv_dir: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.indices = indices.astype(int)
        self.img_root = img_root
        self.image_col = image_col
        self.transform = transform
        self.csv_dir = csv_dir
        # Big enough so RandomResizedCrop / CenterCrop won’t error
        self.fallback_size = max(224, int(fallback_size))

    def __len__(self) -> int:
        return len(self.indices)

    def _resolve_image_path(self, raw_path: str) -> str:
        raw_path = str(raw_path).strip()
        if raw_path == "":
            return ""
        raw_path = raw_path.split("?")[0]

        if os.path.isabs(raw_path) and os.path.isfile(raw_path):
            return raw_path

        cand = os.path.join(self.img_root, raw_path)
        if os.path.isfile(cand):
            return cand

        if self.csv_dir is not None:
            cand2 = os.path.join(self.csv_dir, raw_path)
            if os.path.isfile(cand2):
                return cand2

        base = os.path.basename(raw_path)
        cand3 = os.path.join(self.img_root, base)
        if os.path.isfile(cand3):
            return cand3

        return ""

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        row = self.df.iloc[idx]
        y = int(row["_symmetry_id"])

        img_path = self._resolve_image_path(row[self.image_col])

        # Robust load: missing OR unreadable OR corrupted -> placeholder
        img = None
        if img_path and os.path.isfile(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError):
                img = None

        if img is None:
            img = Image.new("RGB", (self.fallback_size, self.fallback_size), color=(0, 0, 0))

        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long)


# -----------------------------
# Model (image only)
# -----------------------------
class ConvNeXtTinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.25):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = torchvision.models.convnext_tiny(weights=weights)
        in_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0],   # LayerNorm2d
            self.backbone.classifier[1],   # Flatten
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# -----------------------------
# Train / Eval (with progress bars)
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, epoch: int, epochs: int) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    per_cls_correct = np.zeros((4,), dtype=np.int64)
    per_cls_total = np.zeros((4,), dtype=np.int64)

    pbar = tqdm(loader, desc=f"Val   [{epoch}/{epochs}]", leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

        y_np = y.cpu().numpy()
        p_np = pred.cpu().numpy()
        for cls in range(4):
            m = (y_np == cls)
            per_cls_total[cls] += int(m.sum())
            per_cls_correct[cls] += int((p_np[m] == cls).sum())

        pbar.set_postfix(acc=(correct / max(1, total)))

    acc = correct / max(1, total)
    per_cls_acc = per_cls_correct / np.maximum(1, per_cls_total)
    bal_acc = float(per_cls_acc.mean())

    out = {"acc": float(acc), "balanced_acc": float(bal_acc)}
    for cls in range(4):
        out[f"acc_{ID_TO_METRIC[cls]}"] = float(per_cls_acc[cls])
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    grad_clip: float,
    label_smoothing: float,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    pbar = tqdm(loader, desc=f"Train [{epoch}/{epochs}]", leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        pbar.set_postfix(loss=(total_loss / max(1, total_n)))

    return total_loss / max(1, total_n)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Image-only ConvNeXt-Tiny diamond symmetry classifier (4-class).")

    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--img_root", type=str, default=None)
    parser.add_argument(
        "--kaggle_dataset",
        type=str,
        default="aayushpurswani/diamond-images-dataset",
        help="Default dataset used when csv_path/img_root not provided.",
    )
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--image_col", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="symmetry_image_runs")

    args = parser.parse_args()
    seed_everything(args.seed)

    if (args.csv_path is None or args.img_root is None) and args.kaggle_dataset:
        print(f"[AUTO] Downloading dataset via kagglehub: {args.kaggle_dataset}")
        root = maybe_download_with_kagglehub(args.kaggle_dataset)
        picked_csv, picked_img_root = auto_pick_csv_and_img_root(root)
        args.csv_path = picked_csv
        args.img_root = picked_img_root
        print(f"[AUTO] Using CSV: {args.csv_path}")
        print(f"[AUTO] Using img_root: {args.img_root}")

    if args.csv_path is None or args.img_root is None:
        raise SystemExit("You must provide --csv_path and --img_root, OR have kagglehub installed (default mode).")

    csv_path = args.csv_path
    img_root = args.img_root
    csv_dir = os.path.dirname(os.path.abspath(csv_path))

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if len(df) < 20:
        raise SystemExit(f"CSV looks too small ({len(df)} rows). Check: {csv_path}")

    target_col = args.target_col or autodetect_target_col(df)
    image_col = args.image_col or autodetect_image_col(df)

    df = df.copy()
    df["_symmetry_norm"] = df[target_col].map(normalize_symmetry_label)
    df = df[df["_symmetry_norm"].notna()].reset_index(drop=True)
    if len(df) < 20:
        raise SystemExit("Too few rows after filtering invalid symmetry labels.")
    df["_symmetry_id"] = df["_symmetry_norm"].map(SYM_STR_TO_ID).astype(int)

    y = df["_symmetry_id"].to_numpy(dtype=np.int64)
    idx_train, idx_val = stratified_split_indices(y, args.val_frac, args.seed)
    if len(idx_train) == 0 or len(idx_val) == 0:
        raise SystemExit("Train/val split failed. Try changing --val_frac or check label distribution.")

    y_train = df.loc[idx_train, "_symmetry_id"].to_numpy(dtype=np.int64)
    class_counts = np.bincount(y_train, minlength=4).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
    mean, std = get_norm_from_weights(weights)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.70, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.12, 0.12, 0.08, 0.02)], p=0.6),
        transforms.RandomGrayscale(p=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.15)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # placeholder image size: big enough for crops
    fallback_size = int(args.img_size * 1.2)

    ds_train = DiamondSymmetryImageDataset(df, idx_train, img_root, image_col, train_tfms, fallback_size, csv_dir=csv_dir)
    ds_val = DiamondSymmetryImageDataset(df, idx_val, img_root, image_col, val_tfms, fallback_size, csv_dir=csv_dir)

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = ConvNeXtTinyClassifier(num_classes=4, dropout=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    scaler = make_grad_scaler(device)

    with open(os.path.join(run_dir, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"csv_path={csv_path}\n")
        f.write(f"img_root={img_root}\n")
        f.write(f"target_col={target_col}\n")
        f.write(f"image_col={image_col}\n")
        f.write(f"img_size={args.img_size}\n")
        f.write(f"class_counts_train={class_counts.tolist()}\n")
        f.write(f"class_weights_train={class_weights.tolist()}\n")
        f.write(f"mean={mean}\nstd={std}\n")
        f.write(f"device={device}\n")

    best_bal = -1.0
    best_state = None

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "val_acc", "val_bal_acc", "acc_excellent", "acc_very_good", "acc_good", "acc_fair"])

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            epoch=epoch,
            epochs=args.epochs,
        )

        for _ in range(len(train_loader)):
            scheduler.step()

        metrics = evaluate(model, val_loader, device, epoch=epoch, epochs=args.epochs)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={loss:.4f} | "
            f"val_acc={metrics['acc']:.4f} | val_bal_acc={metrics['balanced_acc']:.4f} | "
            f"Ex={metrics['acc_excellent']:.3f} VG={metrics['acc_very_good']:.3f} "
            f"G={metrics['acc_good']:.3f} F={metrics['acc_fair']:.3f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, loss, metrics["acc"], metrics["balanced_acc"],
                metrics["acc_excellent"], metrics["acc_very_good"], metrics["acc_good"], metrics["acc_fair"]
            ])

        if metrics["balanced_acc"] > best_bal:
            best_bal = metrics["balanced_acc"]
            best_state = copy.deepcopy(model.state_dict())
            ckpt_path = os.path.join(run_dir, "best_model.pt")
            torch.save(
                {
                    "model_state": best_state,
                    "sym_mapping": SYM_STR_TO_ID,
                    "target_col": target_col,
                    "image_col": image_col,
                    "img_size": args.img_size,
                    "mean": mean,
                    "std": std,
                    "metrics": metrics,
                    "args": vars(args),
                },
                ckpt_path,
            )

    print(f"\nBest val balanced_acc={best_bal:.4f}")
    print(f"Saved to: {os.path.join(run_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
