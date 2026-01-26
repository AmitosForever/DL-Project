import os
import re
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Repro + speed
# -----------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# -----------------------------
# Column guessing
# -----------------------------
def guess_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols = [c.lower() for c in columns]
    for cand in candidates:
        for i, c in enumerate(cols):
            if cand in c:
                return columns[i]
    return None


# -----------------------------
# Kaggle auto download
# -----------------------------
def auto_download_diamond_dataset() -> str:
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError("kagglehub לא מותקן. התקן: pip install kagglehub") from e
    return kagglehub.dataset_download("aayushpurswani/diamond-images-dataset")


def find_best_csv(ds_root: str) -> str:
    csvs = []
    for r, _, files in os.walk(ds_root):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(r, f))
    if not csvs:
        raise FileNotFoundError(f"לא נמצאו CSV תחת: {ds_root}")

    def score_csv(p: str) -> int:
        try:
            df = pd.read_csv(p, nrows=120)
        except Exception:
            return -999
        cols = [c.lower() for c in df.columns.tolist()]
        s = 0
        if any("polish" in c for c in cols):
            s += 12
        if any("finish" in c for c in cols):
            s += 6
        if any(k in c for c in cols for k in ["image", "img", "path", "file", "filename", "url"]):
            s += 5
        return s

    scored = [(score_csv(p), p) for p in csvs]
    scored.sort(reverse=True, key=lambda x: x[0])
    best_score, best_path = scored[0]
    return best_path if best_score >= 0 else csvs[0]


# -----------------------------
# Paths + image indexing
# -----------------------------
def is_image_file(fn: str) -> bool:
    ext = os.path.splitext(fn)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def build_image_index(root_dir: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for r, _, files in os.walk(root_dir):
        for f in files:
            if is_image_file(f):
                base = os.path.basename(f)
                if base not in idx:
                    idx[base] = os.path.join(r, f)
    return idx


def resolve_image_path(val: str, root_dir: str, image_index: Dict[str, str]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None

    if os.path.isabs(s) and os.path.isfile(s):
        return s

    p = os.path.join(root_dir, s.replace("/", os.sep))
    if os.path.isfile(p):
        return p

    base = os.path.basename(s)
    base = re.split(r"[?#]", base)[0]
    if base in image_index:
        return image_index[base]

    return None


# -----------------------------
# Crop cache (diamond-focused)
# -----------------------------
def crop_center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def crop_diamond_cv2(img: Image.Image) -> Image.Image:
    import cv2
    rgb = np.array(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_center_square(img)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    H, W = gray.shape[:2]
    if w * h < 0.02 * (W * H):
        return crop_center_square(img)

    pad = int(0.08 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    crop = img.crop((x0, y0, x1, y1))
    return crop_center_square(crop)


def get_cropped_path(cache_dir: str, rel_path: str) -> str:
    rel_path = rel_path.replace("/", os.sep)
    base, _ = os.path.splitext(rel_path)
    out_path = os.path.join(cache_dir, base + ".jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def preprocess_crops(
    df: pd.DataFrame,
    img_col: str,
    root_dir: str,
    cache_dir: str,
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)

    has_cv2 = False
    try:
        import cv2  # noqa
        has_cv2 = True
    except Exception:
        has_cv2 = False

    image_index = build_image_index(root_dir)

    new_paths = []
    n = len(df) if max_items is None else min(len(df), max_items)

    pbar = tqdm(range(n), desc="preprocess_crops", leave=True, dynamic_ncols=True)
    bad = 0
    for i in pbar:
        rel = str(df.iloc[i][img_col])
        src = resolve_image_path(rel, root_dir, image_index)
        if src is None:
            bad += 1
            new_paths.append(rel)
            continue

        dst = get_cropped_path(cache_dir, rel)

        if os.path.exists(dst):
            new_paths.append(os.path.relpath(dst, cache_dir))
            continue

        try:
            img = Image.open(src).convert("RGB")
            img = crop_diamond_cv2(img) if has_cv2 else crop_center_square(img)
            img.save(dst, quality=92, optimize=True)
            new_paths.append(os.path.relpath(dst, cache_dir))
        except Exception:
            bad += 1
            new_paths.append(src)

        if (i + 1) % 500 == 0:
            pbar.set_postfix(bad=bad)

    df2 = df.copy()
    df2[img_col] = new_paths + df2[img_col].iloc[len(new_paths):].tolist()
    return df2


# -----------------------------
# Stratified split
# -----------------------------
def stratified_split_indices(labels: List[int], val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    train_idx, val_idx = [], []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n <= 1:
            train_idx.extend(idxs)
            continue
        n_val = int(math.floor(n * val_split))
        n_val = min(n_val, n - 1)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# -----------------------------
# Dataset (Binary EX vs Rest)
# -----------------------------
class PolishBinaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_col: str,
        label_col: str,
        root_dir: str,
        image_index: Dict[str, str],
        positive_is_ex: bool = True,
        tfm=None,
        max_decode_retries: int = 10,
    ):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.root_dir = root_dir
        self.image_index = image_index
        self.positive_is_ex = bool(positive_is_ex)
        self.tfm = tfm
        self.max_decode_retries = max_decode_retries

    def __len__(self):
        return len(self.df)

    def _to_binary(self, s: str) -> Optional[int]:
        s = str(s).strip().upper()
        if s == "EX":
            return 1 if self.positive_is_ex else 0
        if s in ["VG", "GD", "FR", "FAIR", "PO", "PR", "POOR"]:
            return 0 if self.positive_is_ex else 1
        # Unknown labels -> skip
        return None

    def __getitem__(self, idx):
        last_err = None
        for _ in range(self.max_decode_retries):
            row = self.df.iloc[idx]
            p = str(row[self.img_col])
            yb = self._to_binary(row[self.label_col])
            if yb is None:
                idx = torch.randint(0, len(self.df), (1,)).item()
                continue

            img_path = resolve_image_path(p, self.root_dir, self.image_index)
            if img_path is None:
                last_err = (p, "resolve_image_path=None")
                idx = torch.randint(0, len(self.df), (1,)).item()
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                if self.tfm is not None:
                    img = self.tfm(img)
                y = torch.tensor([float(yb)], dtype=torch.float32)  # shape [1]
                return img, y
            except Exception as e:
                last_err = (img_path, repr(e))
                idx = torch.randint(0, len(self.df), (1,)).item()

        raise RuntimeError(f"Decode failed. Last: {last_err[0]} | {last_err[1]}")


# -----------------------------
# Model (ConvNeXt-Tiny) - Binary head
# -----------------------------
class PolishBinaryNet(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)

        in_features = self.backbone.classifier[2].in_features  # 768
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),  # logits for BCE
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # [B]


# -----------------------------
# EMA
# -----------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}


# -----------------------------
# AMP helpers
# -----------------------------
def get_amp(device: torch.device):
    if device.type != "cuda":
        return None, None, False

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.amp.GradScaler(enabled=True)
        return torch.amp.autocast, scaler, True

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    return torch.cuda.amp.autocast, scaler, True


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


def autocast_context(autocast_fn, device: torch.device, enabled: bool):
    if (not enabled) or (autocast_fn is None) or (device.type != "cuda"):
        return _NullCtx()
    try:
        return autocast_fn("cuda", enabled=True)
    except TypeError:
        return autocast_fn(enabled=True)


# -----------------------------
# Transforms (crop-friendly)
# -----------------------------
def build_transforms(img_size: int, mean, std, train: bool, random_erasing: float = 0.0):
    if train:
        tfms = [
            transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0)),  # more aggressive than 0.85
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        if random_erasing and random_erasing > 0:
            tfms.append(transforms.RandomErasing(p=float(random_erasing), scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"))
        return transforms.Compose(tfms)
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# -----------------------------
# Metrics + Confusion
# -----------------------------
@dataclass
class BinMetrics:
    loss: float
    acc: float
    bal_acc: float
    f1: float
    prec: float
    rec: float
    thr: float


def bin_confusion(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # y_true,y_pred in {0,1}, shape [N]
    y_true = y_true.to(torch.int64)
    y_pred = y_pred.to(torch.int64)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return torch.stack([torch.stack([tn, fp]), torch.stack([fn, tp])]).to(torch.int64)


def metrics_from_bin_conf(conf: torch.Tensor) -> Tuple[float, float, float, float, float]:
    # conf = [[tn, fp],[fn,tp]]
    conf = conf.float()
    tn, fp = conf[0, 0], conf[0, 1]
    fn, tp = conf[1, 0], conf[1, 1]
    total = (tn + fp + fn + tp).clamp_min(1.0)
    acc = ((tp + tn) / total).item()

    rec_pos = (tp / (tp + fn + 1e-12)).item()
    rec_neg = (tn / (tn + fp + 1e-12)).item()
    bal_acc = 0.5 * (rec_pos + rec_neg)

    prec = (tp / (tp + fp + 1e-12)).item()
    rec = rec_pos
    f1 = (2 * prec * rec / (prec + rec + 1e-12))
    return acc, bal_acc, float(f1), float(prec), float(rec)


def print_bin_conf(conf: torch.Tensor, title: str):
    conf = conf.detach().cpu().numpy().astype(int)
    tn, fp = conf[0, 0], conf[0, 1]
    fn, tp = conf[1, 0], conf[1, 1]
    print(title)
    print("true\\pred        nonEX        EX")
    print(f"nonEX        {tn:>8d}  {fp:>8d}")
    print(f"EX           {fn:>8d}  {tp:>8d}")


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler_steps: int,
    autocast_fn,
    scaler,
    use_amp: bool,
    ema: Optional[ModelEMA],
    bce: nn.Module,
    grad_clip: float,
    accum_steps: int,
) -> float:
    model.train()

    running = 0.0
    n = 0

    accum_steps = max(1, int(accum_steps))
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc="train", leave=False, dynamic_ncols=True)
    micro = 0
    updates = 0

    for step, (x, y) in pbar:
        micro += 1
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1)  # [B]

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        with autocast_context(autocast_fn, device, enabled=use_amp):
            logits = model(x)  # [B]
            loss = bce(logits, y)

        running += loss.item() * x.size(0)
        n += x.size(0)

        if use_amp:
            (scaler.scale(loss) / accum_steps).backward()
        else:
            (loss / accum_steps).backward()

        do_step = (micro % accum_steps == 0) or (step == len(loader))
        if do_step:
            updates += 1
            if use_amp:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # simple cosine-like schedule by manual lr decay (keep it stable & predictable)
            if scheduler_steps > 0:
                t = min(1.0, updates / float(scheduler_steps))
                lr0 = optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"])
                lr = lr0 * (0.5 * (1.0 + math.cos(math.pi * t)))
                optimizer.param_groups[0]["lr"] = lr

            if ema is not None:
                ema.update(model)

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()
            batch_acc = (pred == y).float().mean().item()

        avg_loss = running / max(1, n)
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{batch_acc*100:.2f}%", lr=f"{lr_now:.2e}", upd=updates, accum=accum_steps)

    return running / max(1, n)


@torch.no_grad()
def evaluate_collect(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bce: nn.Module,
    desc: str = "val",
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits = []
    all_y = []

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1)

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        logits = model(x)
        loss = bce(logits, y)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        all_logits.append(logits.detach())
        all_y.append(y.detach())

        avg_loss = total_loss / max(1, n)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)
    return (total_loss / max(1, n)), all_logits, all_y


def pick_best_threshold(logits: torch.Tensor, y: torch.Tensor, mode: str = "bal_acc") -> Tuple[float, torch.Tensor]:
    # brute force over a grid of thresholds
    probs = torch.sigmoid(logits)
    thresholds = torch.linspace(0.05, 0.95, steps=37, device=probs.device)

    best_thr = 0.5
    best_score = -1.0
    best_conf = None

    y_true = y.round().clamp(0, 1).to(torch.int64)

    for thr in thresholds:
        y_pred = (probs > thr).to(torch.int64)
        conf = bin_confusion(y_true, y_pred)
        acc, bal_acc, f1, prec, rec = metrics_from_bin_conf(conf)

        score = bal_acc if mode == "bal_acc" else f1
        if score > best_score:
            best_score = score
            best_thr = float(thr.item())
            best_conf = conf

    return best_thr, best_conf


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default=None)
    ap.add_argument("--root_dir", default=None)
    ap.add_argument("--img_col", default=None)
    ap.add_argument("--label_col", default=None)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--img_size_ft", type=int, default=512)
    ap.add_argument("--ft_from_epoch", type=int, default=9)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_ft", type=float, default=8e-5)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--effective_batch", type=int, default=64)
    ap.add_argument("--accum_steps", type=int, default=None)

    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--out", default="polish_ex_vs_rest_best.pt")
    ap.add_argument("--out_f1", default="polish_ex_vs_rest_best_f1.pt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_decode_retries", type=int, default=10)

    # Crop cache ON by default
    ap.add_argument("--crop_cache", action="store_true", default=True)
    ap.add_argument("--no_crop_cache", action="store_true")
    ap.add_argument("--cache_dir", default="_polish_cache_crops")

    # Sampler ON by default
    ap.add_argument("--sampler", action="store_true", default=True)
    ap.add_argument("--no_sampler", action="store_true")

    # Loss balancing
    ap.add_argument("--pos_weight_mode", default="ratio", choices=["ratio", "none"])
    ap.add_argument("--focal_gamma", type=float, default=0.0)  # 0 disables focal
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Aug
    ap.add_argument("--random_erasing", type=float, default=0.15)

    # Threshold selection
    ap.add_argument("--thr_mode", default="bal_acc", choices=["bal_acc", "f1"])

    # EMA
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)

    args = ap.parse_args()

    if args.no_crop_cache:
        args.crop_cache = False
    if args.no_sampler:
        args.sampler = False
    if args.no_ema:
        args.ema = False

    if args.accum_steps is None:
        args.accum_steps = max(1, int(math.ceil(args.effective_batch / max(1, args.batch))))

    seed_all(args.seed)

    # Auto-download
    if args.csv is None:
        ds_root = auto_download_diamond_dataset()
        args.csv = find_best_csv(ds_root)
        if args.root_dir is None:
            args.root_dir = ds_root
        print(f"[AUTO] Downloaded dataset root: {ds_root}")
        print(f"[AUTO] Using CSV: {args.csv}")
        print(f"[AUTO] Using root_dir: {args.root_dir}")

    df = pd.read_csv(args.csv)
    cols = df.columns.tolist()

    img_col = args.img_col or guess_column(cols, ["image", "img", "path", "file", "filename", "url"])
    label_col = args.label_col or guess_column(cols, ["polish", "finish"])

    if img_col is None or label_col is None:
        raise ValueError(f"לא הצלחתי לזהות עמודות. columns={cols}. תן --img_col ו--label_col ידנית.")

    df = df.dropna(subset=[img_col, label_col]).copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()

    # Keep only EX/VG/GD (and map others to nonEX if present)
    # We'll allow VG/GD as nonEX. If other labels exist, they become nonEX by default.
    def to_bin(s: str) -> Optional[int]:
        s = str(s).strip().upper()
        if s == "EX":
            return 1
        if s in ["VG", "GD", "FR", "FAIR", "PO", "PR", "POOR"]:
            return 0
        # If dataset has only EX/VG/GD, unknowns are likely garbage -> skip
        # but you can change this if needed
        return None

    df["_ybin"] = df[label_col].map(to_bin)
    df = df.dropna(subset=["_ybin"]).copy()
    df["_ybin"] = df["_ybin"].astype(int)

    # Crop cache
    if args.crop_cache:
        cache_root = os.path.join(os.path.dirname(args.csv), args.cache_dir)
        print(f"[INFO] Crop cache: ON -> {cache_root}")
        df = preprocess_crops(df, img_col=img_col, root_dir=args.root_dir, cache_dir=cache_root)
        args.root_dir = cache_root
    else:
        print("[INFO] Crop cache: OFF")

    # Summary
    n_total = len(df)
    n_pos = int(df["_ybin"].sum())
    n_neg = int(n_total - n_pos)
    print(f"[INFO] img_col={img_col} label_col={label_col} task=EX_vs_rest (EX=1, nonEX=0)")
    print(f"[INFO] n_total={n_total} | EX(pos)={n_pos} | nonEX(neg)={n_neg}")
    print(f"[INFO] GradAccum: batch={args.batch} effective_batch={args.effective_batch} accum_steps={args.accum_steps}")
    print(f"[INFO] Sampler: {'ON' if args.sampler else 'OFF'}")

    # Split (stratified on binary)
    train_idx, val_idx = stratified_split_indices(df["_ybin"].tolist(), args.val_split, args.seed)
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # Image index
    image_index = build_image_index(args.root_dir)

    # Transforms
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std
    tfm_train = build_transforms(args.img_size, mean, std, train=True, random_erasing=args.random_erasing)
    tfm_val = build_transforms(args.img_size_ft, mean, std, train=False, random_erasing=0.0)

    ds_train = PolishBinaryDataset(
        df_train, img_col, label_col, args.root_dir, image_index,
        positive_is_ex=True, tfm=tfm_train, max_decode_retries=args.max_decode_retries
    )
    ds_val = PolishBinaryDataset(
        df_val, img_col, label_col, args.root_dir, image_index,
        positive_is_ex=True, tfm=tfm_val, max_decode_retries=args.max_decode_retries
    )

    # Sampler (binary balanced)
    sampler = None
    if args.sampler:
        y_train = df_train["_ybin"].astype(int).tolist()
        cnt0 = max(1, sum(1 for y in y_train if y == 0))
        cnt1 = max(1, sum(1 for y in y_train if y == 1))
        w0 = 1.0 / cnt0
        w1 = 1.0 / cnt1
        sample_weights = [w1 if y == 1 else w0 for y in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"[INFO] Sampler weights: w_nonEX={w0:.6f} w_EX={w1:.6f}")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Device + model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = PolishBinaryNet(dropout=args.dropout).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Optimizer (head LR multiplier)
    # ConvNeXt structure: backbone.features + classifier
    head_params = []
    body_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.classifier" in name:
            head_params.append(p)
        else:
            body_params.append(p)

    head_lr_mult = 3.0
    print(f"[INFO] head_lr_mult={head_lr_mult} | random_erasing={args.random_erasing}")

    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": args.lr, "weight_decay": args.wd, "initial_lr": args.lr},
            {"params": head_params, "lr": args.lr * head_lr_mult, "weight_decay": args.wd, "initial_lr": args.lr * head_lr_mult},
        ]
    )

    # AMP
    autocast_fn, scaler, amp_available = get_amp(device)
    use_amp = amp_available and (not args.no_amp)

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # pos_weight for BCE (to care about minority)
    # Note: EX is majority here. But still, you may want to favor nonEX detection.
    # We're classifying EX as positive. If you want "nonEX is positive", flip labels.
    # For EX positive: pos_weight = neg/pos will *increase* EX importance (not what you want).
    # So we default to none. Balance is handled by sampler.
    pos_weight = None
    if args.pos_weight_mode == "ratio":
        # Usually you'd use this when positive is minority. Here EX is majority, so skip.
        # We'll still compute and print it for transparency.
        ytr = df_train["_ybin"].astype(int).values
        pos = max(1, int(ytr.sum()))
        neg = max(1, int((1 - ytr).sum()))
        pw = float(neg / pos)
        print(f"[INFO] pos_weight ratio (neg/pos) = {pw:.6f} (NOTE: EX is majority, sampler does the heavy lifting)")
        # keep disabled by default to avoid making EX even stronger
        pos_weight = None
    else:
        print("[INFO] pos_weight: OFF")

    # Loss
    bce_base = nn.BCEWithLogitsLoss(pos_weight=(torch.tensor([pos_weight], device=device) if pos_weight is not None else None))

    def focal_bce_with_logits(logits, targets, gamma: float):
        # targets in {0,1}
        base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        w = (1 - pt).pow(gamma)
        return (w * base).mean()

    best_bal = -1.0
    best_f1 = -1.0

    # scheduler_steps: total optimizer updates (rough)
    updates_per_epoch = int(math.ceil(len(dl_train) / max(1, args.accum_steps)))
    total_updates = updates_per_epoch * max(1, args.epochs)
    print(f"[INFO] updates_per_epoch={updates_per_epoch} | total_updates={total_updates}")

    for epoch in range(1, args.epochs + 1):
        # Fine-tune stage: larger img + smaller LR
        if epoch == args.ft_from_epoch:
            print(f"[STAGE] Fine-tune from epoch {epoch}: img_size={args.img_size_ft}, lr_ft={args.lr_ft}")
            ds_train.tfm = build_transforms(args.img_size_ft, mean, std, train=True, random_erasing=max(0.0, args.random_erasing * 0.7))
            # set lrs
            for pg in optimizer.param_groups:
                base = pg.get("initial_lr", pg["lr"])
                # scale down proportionally
                mult = (base / args.lr) if args.lr > 0 else 1.0
                pg["lr"] = args.lr_ft * mult

        # choose loss wrapper
        if args.focal_gamma and args.focal_gamma > 0:
            def bce(logits, y):  # closure
                return focal_bce_with_logits(logits, y, gamma=float(args.focal_gamma))
            print(f"[INFO] Focal gamma={args.focal_gamma}")
        else:
            bce = bce_base

        train_loss = train_one_epoch(
            model=model,
            loader=dl_train,
            device=device,
            optimizer=optimizer,
            scheduler_steps=total_updates,
            autocast_fn=autocast_fn,
            scaler=scaler,
            use_amp=use_amp,
            ema=ema,
            bce=bce,
            grad_clip=args.grad_clip,
            accum_steps=args.accum_steps,
        )

        # Validate with EMA weights if enabled
        if ema is not None:
            ema.apply_shadow(model)

        val_loss, val_logits, val_y = evaluate_collect(model, dl_val, device, bce=(bce_base if not callable(bce_base) else bce_base), desc="val")

        # pick best threshold on val
        best_thr, best_conf = pick_best_threshold(val_logits, val_y, mode=args.thr_mode)
        acc, bal_acc, f1, prec, rec = metrics_from_bin_conf(best_conf)

        if ema is not None:
            ema.restore(model)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"thr*={best_thr:.2f} | val_acc={acc*100:.2f}% | val_bal_acc={bal_acc*100:.2f}% | "
            f"val_F1={f1:.4f} | val_P={prec*100:.2f}% | val_R={rec*100:.2f}%"
        )

        print_bin_conf(best_conf, title=f"[VAL] Confusion matrix (thr={best_thr:.2f}) @ epoch {epoch:02d}")

        # Save best by bal_acc
        if bal_acc > best_bal:
            best_bal = bal_acc
            ckpt = {
                "model": model.state_dict(),
                "ema": (ema.shadow if ema is not None else None),
                "task": "polish_EX_vs_rest_binary",
                "threshold": best_thr,
                "img_col": img_col,
                "label_col": label_col,
                "root_dir": args.root_dir,
                "backbone": "convnext_tiny",
                "weights": "ConvNeXt_Tiny_Weights.DEFAULT",
                "train": {
                    "epochs": args.epochs,
                    "img_size": args.img_size,
                    "img_size_ft": args.img_size_ft,
                    "ft_from_epoch": args.ft_from_epoch,
                    "lr": args.lr,
                    "lr_ft": args.lr_ft,
                    "wd": args.wd,
                    "batch": args.batch,
                    "effective_batch": args.effective_batch,
                    "accum_steps": args.accum_steps,
                    "sampler": args.sampler,
                    "pos_weight_mode": args.pos_weight_mode,
                    "focal_gamma": args.focal_gamma,
                    "random_erasing": args.random_erasing,
                    "thr_mode": args.thr_mode,
                    "ema_decay": (args.ema_decay if args.ema else None),
                },
                "best": {
                    "bal_acc": best_bal,
                    "acc": acc,
                    "f1": f1,
                    "prec": prec,
                    "rec": rec,
                }
            }
            torch.save(ckpt, args.out)
            print(f"[SAVE] best_bal -> {args.out} (val_bal_acc={best_bal*100:.2f}%)")

        # Save best by F1 too (often more meaningful)
        if f1 > best_f1:
            best_f1 = f1
            ckpt2 = {
                "model": model.state_dict(),
                "ema": (ema.shadow if ema is not None else None),
                "task": "polish_EX_vs_rest_binary",
                "threshold": best_thr,
                "root_dir": args.root_dir,
                "best_f1": best_f1,
            }
            torch.save(ckpt2, args.out_f1)
            print(f"[SAVE] best_f1 -> {args.out_f1} (val_F1={best_f1:.4f})")

    print(f"[DONE] best_bal_acc={best_bal*100:.2f}% | best_f1={best_f1:.4f} | saved={args.out} / {args.out_f1}")


if __name__ == "__main__":
    main()
