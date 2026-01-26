import os
import re
import math
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
from torchvision.models import resnet18, ResNet18_Weights

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
# Target transform
# -----------------------------
class Log1pTarget:
    @staticmethod
    def forward(y: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(y, min=0.0))

    @staticmethod
    def inverse(y_t: torch.Tensor) -> torch.Tensor:
        return torch.expm1(y_t)


# -----------------------------
# Dataset
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
# Model
# -----------------------------
class CaratNet(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.weights = weights
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x)


def set_trainable_backbone(model: CaratNet, train_backbone: bool):
    # fc always trainable
    for name, p in model.named_parameters():
        if name.startswith("backbone.fc"):
            p.requires_grad = True
        else:
            p.requires_grad = bool(train_backbone)


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
# Scheduler
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
# Transforms (fast)
# -----------------------------
def build_transforms(img_size: int, mean, std, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.92, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.03, hue=0.01),
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
    target_norm: bool,
    y_mean_t: float,
    y_std_t: float,
    huber_delta: float,
    grad_clip: float,
    accum_steps: int,
) -> float:
    model.train()
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

        y_t = Log1pTarget.forward(y)
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
    loss_sum = nn.SmoothL1Loss(beta=huber_delta, reduction="sum")

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
            loss = loss_sum(pred_tn, y_tn)
            pred_t = pred_tn * (y_std_t + 1e-12) + y_mean_t
        else:
            loss = loss_sum(pred_tn, y_t)
            pred_t = pred_tn

        pred = Log1pTarget.inverse(pred_t)

        res = pred - y
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

    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--warmup_epochs", type=int, default=1)

    ap.add_argument("--img_size", type=int, default=256)

    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lr_backbone", type=float, default=8e-5)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--effective_batch", type=int, default=256)
    ap.add_argument("--accum_steps", type=int, default=None)

    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--out", default="carat_fast_best.pt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_decode_retries", type=int, default=10)

    ap.add_argument("--sampler", action="store_true", default=True)
    ap.add_argument("--no_sampler", action="store_true")
    ap.add_argument("--sampler_bins", type=int, default=20)

    ap.add_argument("--target_norm", action="store_true", default=True)
    ap.add_argument("--no_target_norm", action="store_true")

    ap.add_argument("--huber_delta", type=float, default=0.20)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--freeze_epochs", type=int, default=2)  # head-only warm start

    args = ap.parse_args()

    if args.no_sampler:
        args.sampler = False
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

    y_all = df[y_col].astype(np.float64).values
    print(f"[INFO] img_col={img_col} y_col={y_col} N={len(df)}")
    print(
        f"[INFO] y stats: mean={y_all.mean():.4f} std={y_all.std():.4f} "
        f"min={y_all.min():.4f} p50={np.median(y_all):.4f} max={y_all.max():.4f}"
    )
    print(f"[INFO] GradAccum: batch={args.batch} effective_batch={args.effective_batch} accum_steps={args.accum_steps}")

    train_idx, val_idx = stratified_split_regression(
        y_all, args.val_split, args.seed, n_bins=max(5, int(args.sampler_bins))
    )
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # log target stats
    y_train = torch.tensor(df_train[y_col].astype(np.float32).values)
    y_train_t = Log1pTarget.forward(y_train)
    y_mean_t = float(y_train_t.mean().item())
    y_std_t = float(y_train_t.std(unbiased=False).item())
    if y_std_t <= 0:
        y_std_t = 1.0
    print(f"[INFO] target=log1p(carat) | target_norm={'ON' if args.target_norm else 'OFF'} | y_mean_t={y_mean_t:.6f} y_std_t={y_std_t:.6f}")

    image_index = build_image_index(args.root_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = CaratNet(dropout=args.dropout).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    weights = ResNet18_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std

    tfm_train = build_transforms(args.img_size, mean, std, train=True)
    tfm_val = build_transforms(args.img_size, mean, std, train=False)

    ds_train = CaratDataset(df_train, img_col, y_col, args.root_dir, image_index, tfm=tfm_train, max_decode_retries=args.max_decode_retries)
    ds_val = CaratDataset(df_val, img_col, y_col, args.root_dir, image_index, tfm=tfm_val, max_decode_retries=args.max_decode_retries)

    sampler = None
    if args.sampler:
        ytr = df_train[y_col].astype(np.float64).values
        n_bins = max(5, int(args.sampler_bins))
        qs = np.linspace(0, 1, n_bins + 1)  # <- fixed below
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

    # Freeze backbone initially
    set_trainable_backbone(model, train_backbone=False)

    # Optimizer with two groups
    head_params = list(model.backbone.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and (not n.startswith("backbone.fc"))]

    # when backbone frozen, backbone_params is empty; fine
    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)

    autocast_fn, scaler, amp_available = get_amp(device)
    use_amp = amp_available and (not args.no_amp)

    updates_per_epoch = int(math.ceil(len(dl_train) / max(1, args.accum_steps)))
    warmup_steps = updates_per_epoch * max(0, int(args.warmup_epochs))
    total_steps = updates_per_epoch * max(1, int(args.epochs))
    sched = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    sched.set_lr(0)

    best_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            print(f"[STAGE] Unfreeze backbone at epoch {epoch}")
            set_trainable_backbone(model, train_backbone=True)

            head_params = list(model.backbone.fc.parameters())
            backbone_params = [p for n, p in model.named_parameters() if (not n.startswith("backbone.fc")) and p.requires_grad]
            optimizer = torch.optim.AdamW(
                [{"params": backbone_params, "lr": args.lr_backbone},
                 {"params": head_params, "lr": args.lr_head}],
                weight_decay=args.wd
            )
            sched = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
            sched.set_lr(sched.step_num)

        train_loss = train_one_epoch(
            model=model,
            loader=dl_train,
            device=device,
            optimizer=optimizer,
            scheduler=sched,
            autocast_fn=autocast_fn,
            scaler=scaler,
            use_amp=use_amp,
            target_norm=args.target_norm,
            y_mean_t=y_mean_t,
            y_std_t=y_std_t,
            huber_delta=args.huber_delta,
            grad_clip=args.grad_clip,
            accum_steps=args.accum_steps,
        )

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

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics.loss:.6f} | "
            f"val_MAE={val_metrics.mae:.5f} | "
            f"val_RMSE={val_metrics.rmse:.5f} | "
            f"val_R2={val_metrics.r2:.4f}"
        )

        if val_metrics.rmse < best_rmse:
            best_rmse = val_metrics.rmse
            ckpt = {
                "model": model.state_dict(),
                "img_col": img_col,
                "y_col": y_col,
                "root_dir": args.root_dir,
                "task": "carat_regression_fast_resnet18_log1p",
                "target_norm": args.target_norm,
                "y_mean_t": y_mean_t,
                "y_std_t": y_std_t,
                "best": {"val_rmse": best_rmse},
                "train": vars(args),
            }
            torch.save(ckpt, args.out)
            print(f"[SAVE] best -> {args.out} (val_RMSE={best_rmse:.5f})")

    print(f"[DONE] best_val_RMSE={best_rmse:.5f} | saved={args.out}")


if __name__ == "__main__":
    main()
