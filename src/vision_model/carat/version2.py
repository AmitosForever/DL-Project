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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    convnext_small,
    ConvNeXt_Small_Weights,
)

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
        if any("carat" in c for c in cols):
            s += 12
        if any(k in c for c in cols for k in ["image", "img", "path", "file", "filename", "url"]):
            s += 5
        for k in ["cut", "color", "clarity", "polish", "symmetry", "shape"]:
            if any(k in c for c in cols):
                s += 1
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

    pad = int(0.10 * max(w, h))
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
            new_paths.append(rel)

        if (i + 1) % 500 == 0:
            pbar.set_postfix(bad=bad)

    df2 = df.copy()
    df2[img_col] = new_paths + df2[img_col].iloc[len(new_paths):].tolist()
    return df2


# -----------------------------
# Stratified-ish split for regression: bin y, then stratify bins
# -----------------------------
def stratified_split_regression(
    y: np.ndarray,
    val_split: float,
    seed: int,
    n_bins: int = 20
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    y = np.asarray(y, dtype=np.float64)

    # clip extremes for stable binning (not training)
    lo = np.nanpercentile(y, 0.5)
    hi = np.nanpercentile(y, 99.5)
    yc = np.clip(y, lo, hi)

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(yc, qs))
    if len(edges) < 3:
        edges = np.linspace(float(yc.min()), float(yc.max()) + 1e-9, n_bins + 1)

    bins = np.digitize(yc, edges[1:-1], right=False)

    by_bin: Dict[int, List[int]] = {}
    for i, b in enumerate(bins.tolist()):
        by_bin.setdefault(int(b), []).append(i)

    train_idx, val_idx = [], []
    for _, idxs in by_bin.items():
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
# Target transforms (key upgrade)
# -----------------------------
class Log1pTarget:
    @staticmethod
    def forward(y: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(y, min=0.0))

    @staticmethod
    def inverse(y_t: torch.Tensor) -> torch.Tensor:
        return torch.expm1(y_t)


# -----------------------------
# Dataset (regression)
# -----------------------------
class CaratDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_col: str,
        y_col: str,
        root_dir: str,
        image_index: Dict[str, str],
        tfm=None,
        max_decode_retries: int = 10,
    ):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.y_col = y_col
        self.root_dir = root_dir
        self.image_index = image_index
        self.tfm = tfm
        self.max_decode_retries = max_decode_retries
        self.y = self.df[self.y_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        last_err = None
        for _ in range(self.max_decode_retries):
            row = self.df.iloc[idx]
            p = str(row[self.img_col])

            img_path = resolve_image_path(p, self.root_dir, self.image_index)
            if img_path is None:
                last_err = (p, "resolve_image_path=None")
                idx = torch.randint(0, len(self.df), (1,)).item()
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                if self.tfm is not None:
                    img = self.tfm(img)
                y = torch.tensor(self.y[idx], dtype=torch.float32)
                return img, y
            except Exception as e:
                last_err = (img_path, repr(e))
                idx = torch.randint(0, len(self.df), (1,)).item()

        raise RuntimeError(f"Decode failed. Last: {last_err[0]} | {last_err[1]}")


# -----------------------------
# Models
# -----------------------------
class CaratNet(nn.Module):
    def __init__(self, backbone: str = "effv2s", dropout: float = 0.2):
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name in ["effv2s", "efficientnet_v2_s", "efficientnetv2s"]:
            weights = EfficientNet_V2_S_Weights.DEFAULT
            self.backbone = efficientnet_v2_s(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, 1),
            )
            self._weights = weights

        elif self.backbone_name in ["convnext_small", "convnexts", "cnexts"]:
            weights = ConvNeXt_Small_Weights.DEFAULT
            self.backbone = convnext_small(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(in_features, eps=1e-6),
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1),
            )
            self._weights = weights
        else:
            raise ValueError("backbone לא מוכר. תשתמש ב: effv2s / convnext_small")

    @property
    def weights(self):
        return self._weights

    def forward(self, x):
        return self.backbone(x)


def freeze_backbone(model: CaratNet, freeze: bool):
    # Freeze everything except last classifier block
    for name, p in model.named_parameters():
        p.requires_grad = True

    if not freeze:
        return

    # freeze backbone "features" only
    for name, p in model.named_parameters():
        # keep classifier trainable
        if ".classifier" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


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
# AMP
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
# Scheduler (warmup + cosine), per optimizer update
# -----------------------------
class WarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_num = 0

    def _lr_mult(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return 1.0
        progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def set_lr(self, step: int):
        m = self._lr_mult(step)
        for base, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = base * m

    def step(self):
        self.step_num += 1
        self.set_lr(self.step_num)


# -----------------------------
# Transforms (less "destructive" for regression)
# -----------------------------
def build_transforms(img_size: int, mean, std, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(6),
            transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.04, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# -----------------------------
# Metrics
# -----------------------------
@dataclass
class RegMetrics:
    loss: float
    mae: float
    rmse: float
    r2: float


def safe_r2(sum_y: float, sum_y2: float, sum_res2: float, n: int) -> float:
    if n <= 1:
        return float("nan")
    denom = (sum_y2 - (sum_y * sum_y) / float(n))
    if denom <= 1e-12:
        return float("nan")
    return 1.0 - (sum_res2 / denom)


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosine,
    autocast_fn,
    scaler,
    use_amp: bool,
    ema: Optional[ModelEMA],
    target_norm: bool,
    y_mean_t: float,
    y_std_t: float,
    huber_delta: float,
    grad_clip: float,
    accum_steps: int,
) -> float:
    model.train()

    # Huber is usually better than pure MSE here
    loss_fn = nn.SmoothL1Loss(beta=huber_delta)

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
        y = y.to(device, non_blocking=True)

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        # log1p target
        y_t = Log1pTarget.forward(y)

        # normalize target in log-space (optional, usually helps)
        if target_norm:
            y_tn = (y_t - y_mean_t) / (y_std_t + 1e-12)
        else:
            y_tn = y_t

        with autocast_context(autocast_fn, device, enabled=use_amp):
            pred_tn = model(x).squeeze(1)
            loss = loss_fn(pred_tn, y_tn)

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
            scheduler.step()

            if ema is not None:
                ema.update(model)

        avg_loss = running / max(1, n)
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg_loss:.5f}", lr=f"{lr_now:.2e}", upd=updates, accum=accum_steps)

    return running / max(1, n)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_norm: bool,
    y_mean_t: float,
    y_std_t: float,
    huber_delta: float,
    desc: str = "val",
) -> RegMetrics:
    model.eval()

    loss_fn_sum = nn.SmoothL1Loss(beta=huber_delta, reduction="sum")

    sum_loss = 0.0
    sum_abs = 0.0
    sum_res2 = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        y_t = Log1pTarget.forward(y)

        pred_tn = model(x).squeeze(1)

        if target_norm:
            y_tn = (y_t - y_mean_t) / (y_std_t + 1e-12)
            loss = loss_fn_sum(pred_tn, y_tn)

            # unnormalize back to log-space
            pred_t = pred_tn * (y_std_t + 1e-12) + y_mean_t
        else:
            loss = loss_fn_sum(pred_tn, y_t)
            pred_t = pred_tn

        # back to original carat space
        pred = Log1pTarget.inverse(pred_t)

        res = (pred - y)
        sum_loss += float(loss.item())
        sum_abs += float(res.abs().sum().item())
        sum_res2 += float((res * res).sum().item())
        sum_y += float(y.sum().item())
        sum_y2 += float((y * y).sum().item())
        n += int(y.numel())

        mae = sum_abs / max(1, n)
        rmse = math.sqrt(sum_res2 / max(1, n))
        r2 = safe_r2(sum_y, sum_y2, sum_res2, n)
        pbar.set_postfix(mae=f"{mae:.4f}", rmse=f"{rmse:.4f}", r2=f"{r2:.3f}")

    loss_avg = sum_loss / max(1, n)
    mae = sum_abs / max(1, n)
    rmse = math.sqrt(sum_res2 / max(1, n))
    r2 = safe_r2(sum_y, sum_y2, sum_res2, n)
    return RegMetrics(loss=loss_avg, mae=mae, rmse=rmse, r2=r2)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default=None)
    ap.add_argument("--root_dir", default=None)
    ap.add_argument("--img_col", default=None)
    ap.add_argument("--y_col", default=None)

    ap.add_argument("--backbone", default="effv2s", choices=["effv2s", "convnext_small"])

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup_epochs", type=int, default=2)

    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--img_size_ft", type=int, default=512)
    ap.add_argument("--ft_from_epoch", type=int, default=8)

    ap.add_argument("--freeze_epochs", type=int, default=2)  # train head only first epochs

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_ft", type=float, default=8e-5)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--effective_batch", type=int, default=64)
    ap.add_argument("--accum_steps", type=int, default=None)

    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--out", default="carat_best.pt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_decode_retries", type=int, default=10)

    ap.add_argument("--crop_cache", action="store_true", default=True)
    ap.add_argument("--no_crop_cache", action="store_true")
    ap.add_argument("--cache_dir", default="_carat_cache_crops_v2")

    ap.add_argument("--sampler", action="store_true", default=True)
    ap.add_argument("--no_sampler", action="store_true")
    ap.add_argument("--sampler_bins", type=int, default=20)

    ap.add_argument("--target_norm", action="store_true", default=True)
    ap.add_argument("--no_target_norm", action="store_true")

    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)

    ap.add_argument("--huber_delta", type=float, default=0.20)  # in normalized/log space
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--patience", type=int, default=6)  # early stopping on RMSE

    args = ap.parse_args()

    if args.no_crop_cache:
        args.crop_cache = False
    if args.no_sampler:
        args.sampler = False
    if args.no_ema:
        args.ema = False
    if args.no_target_norm:
        args.target_norm = False

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
    y_col = args.y_col or guess_column(cols, ["carat"])

    if img_col is None or y_col is None:
        raise ValueError(f"לא הצלחתי לזהות עמודות. columns={cols}. תן --img_col ו--y_col ידנית.")

    df = df.copy()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[img_col, y_col]).copy()

    # Crop cache
    if args.crop_cache:
        cache_root = os.path.join(os.path.dirname(args.csv), args.cache_dir)
        print(f"[INFO] Crop cache: ON -> {cache_root}")
        df = preprocess_crops(df, img_col=img_col, root_dir=args.root_dir, cache_dir=cache_root)
        args.root_dir = cache_root
    else:
        print("[INFO] Crop cache: OFF")

    y_all = df[y_col].astype(np.float64).values
    print(f"[INFO] img_col={img_col} y_col={y_col} N={len(df)}")
    print(
        f"[INFO] y stats: mean={y_all.mean():.4f} std={y_all.std():.4f} "
        f"min={y_all.min():.4f} p50={np.median(y_all):.4f} max={y_all.max():.4f}"
    )
    print(f"[INFO] GradAccum: batch={args.batch} effective_batch={args.effective_batch} accum_steps={args.accum_steps}")

    # Split
    train_idx, val_idx = stratified_split_regression(
        y_all, args.val_split, args.seed, n_bins=max(5, int(args.sampler_bins))
    )
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # target stats in LOG space
    y_train = torch.tensor(df_train[y_col].astype(np.float32).values)
    y_train_t = Log1pTarget.forward(y_train)
    y_mean_t = float(y_train_t.mean().item())
    y_std_t = float(y_train_t.std(unbiased=False).item())
    if y_std_t <= 0:
        y_std_t = 1.0
    print(f"[INFO] target: log1p(carat) | target_norm={'ON' if args.target_norm else 'OFF'} | "
          f"y_mean_t={y_mean_t:.6f} y_std_t={y_std_t:.6f}")

    image_index = build_image_index(args.root_dir)

    # Device + model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = CaratNet(backbone=args.backbone, dropout=args.dropout).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    weights = model.weights
    mean, std = weights.transforms().mean, weights.transforms().std

    tfm_train = build_transforms(args.img_size, mean, std, train=True)
    tfm_val = build_transforms(args.img_size_ft, mean, std, train=False)

    ds_train = CaratDataset(df_train, img_col, y_col, args.root_dir, image_index, tfm=tfm_train, max_decode_retries=args.max_decode_retries)
    ds_val = CaratDataset(df_val, img_col, y_col, args.root_dir, image_index, tfm=tfm_val, max_decode_retries=args.max_decode_retries)

    # Sampler by y bins (in original space is fine)
    sampler = None
    if args.sampler:
        ytr = df_train[y_col].astype(np.float64).values
        n_bins = max(5, int(args.sampler_bins))
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(ytr, qs))
        if len(edges) < 3:
            edges = np.linspace(float(ytr.min()), float(ytr.max()) + 1e-9, n_bins + 1)
        bins = np.digitize(ytr, edges[1:-1], right=False)

        counts: Dict[int, int] = {}
        for b in bins.tolist():
            counts[int(b)] = counts.get(int(b), 0) + 1
        sample_weights = [1.0 / counts[int(b)] for b in bins.tolist()]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"[INFO] Sampler: WeightedRandomSampler ON (bins={n_bins})")
    else:
        print("[INFO] Sampler: OFF")

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

    # Freeze first epochs (head-only)
    freeze_backbone(model, freeze=(args.freeze_epochs > 0))

    # Optimizer with smaller LR for backbone and larger for head (simple & effective)
    head_params = []
    base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".classifier" in name:
            head_params.append(p)
        else:
            base_params.append(p)

    # If frozen, base_params might be empty
    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr * 0.35})
    param_groups.append({"params": head_params, "lr": args.lr})

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)

    # AMP
    autocast_fn, scaler, amp_available = get_amp(device)
    use_amp = amp_available and (not args.no_amp)

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # Scheduler: per optimizer update
    updates_per_epoch = int(math.ceil(len(dl_train) / max(1, args.accum_steps)))
    warmup_steps = updates_per_epoch * max(0, int(args.warmup_epochs))
    total_steps = updates_per_epoch * max(1, int(args.epochs))
    sched = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    sched.set_lr(0)

    best_rmse = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        # unfreeze after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            freeze_backbone(model, freeze=False)
            # rebuild optimizer param groups after unfreeze (so backbone trains)
            head_params = []
            base_params = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if ".classifier" in name:
                    head_params.append(p)
                else:
                    base_params.append(p)

            param_groups = [{"params": base_params, "lr": args.lr * 0.35}, {"params": head_params, "lr": args.lr}]
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)

            # reset scheduler with same total plan
            sched = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
            sched.set_lr(sched.step_num)

            print(f"[STAGE] Unfreeze backbone at epoch {epoch}")

        # fine-tune stage: bigger images + lower lr
        if epoch == args.ft_from_epoch:
            print(f"[STAGE] Fine-tune from epoch {epoch}: img_size={args.img_size_ft}, lr={args.lr_ft}")
            ds_train.tfm = build_transforms(args.img_size_ft, mean, std, train=True)
            for pg in optimizer.param_groups:
                # keep head higher if you want, but not too crazy
                pg["lr"] = args.lr_ft if pg is optimizer.param_groups[0] else args.lr_ft * 1.5

        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=dl_train,
            device=device,
            optimizer=optimizer,
            scheduler=sched,
            autocast_fn=autocast_fn,
            scaler=scaler,
            use_amp=use_amp,
            ema=ema,
            target_norm=args.target_norm,
            y_mean_t=y_mean_t,
            y_std_t=y_std_t,
            huber_delta=args.huber_delta,
            grad_clip=args.grad_clip,
            accum_steps=args.accum_steps,
        )

        if ema is not None:
            ema.apply_shadow(model)

        val_metrics = evaluate(
            model=model,
            loader=dl_val,
            device=device,
            target_norm=args.target_norm,
            y_mean_t=y_mean_t,
            y_std_t=y_std_t,
            huber_delta=args.huber_delta,
            desc="val",
        )

        if ema is not None:
            ema.restore(model)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics.loss:.6f} | "
            f"val_MAE={val_metrics.mae:.5f} | "
            f"val_RMSE={val_metrics.rmse:.5f} | "
            f"val_R2={val_metrics.r2:.4f} | "
            f"sec={dt:.1f}"
        )

        improved = val_metrics.rmse < best_rmse - 1e-6
        if improved:
            best_rmse = val_metrics.rmse
            bad_epochs = 0
            ckpt = {
                "model": model.state_dict(),
                "ema": (ema.shadow if ema is not None else None),
                "img_col": img_col,
                "y_col": y_col,
                "root_dir": args.root_dir,
                "backbone": args.backbone,
                "weights": str(type(model.weights).__name__),
                "task": "carat_regression_log1p_huber",
                "target_norm": args.target_norm,
                "y_mean_t": y_mean_t,
                "y_std_t": y_std_t,
                "train": vars(args),
                "best": {"val_rmse": best_rmse},
            }
            torch.save(ckpt, args.out)
            print(f"[SAVE] best -> {args.out} (val_RMSE={best_rmse:.5f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[EARLY STOP] no improvement for {args.patience} epochs. best_RMSE={best_rmse:.5f}")
                break

    print(f"[DONE] best_val_RMSE={best_rmse:.5f} | saved={args.out}")


if __name__ == "__main__":
    main()
