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
        if any("cut" in c for c in cols):
            s += 2
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
def stratified_split_indices(labels: List[str], val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class: Dict[str, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)

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
# Dataset
# -----------------------------
class PolishDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_col: str,
        label_col: str,
        root_dir: str,
        label2idx: Dict[str, int],
        image_index: Dict[str, str],
        tfm=None,
        max_decode_retries: int = 10,
    ):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.root_dir = root_dir
        self.label2idx = label2idx
        self.image_index = image_index
        self.tfm = tfm
        self.max_decode_retries = max_decode_retries

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        last_err = None
        for _ in range(self.max_decode_retries):
            row = self.df.iloc[idx]
            p = str(row[self.img_col])
            y_str = str(row[self.label_col]).strip()

            if y_str not in self.label2idx:
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
                y = self.label2idx[y_str]
                return img, y
            except Exception as e:
                last_err = (img_path, repr(e))
                idx = torch.randint(0, len(self.df), (1,)).item()

        raise RuntimeError(f"Decode failed. Last: {last_err[0]} | {last_err[1]}")


# -----------------------------
# Model (ConvNeXt-Tiny)
# -----------------------------
class PolishNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)

        in_features = self.backbone.classifier[2].in_features  # 768
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


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
# MixUp / CutMix
# -----------------------------
def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=num_classes).float()


def rand_bbox(W: int, H: int, lam: float):
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def apply_mixup_cutmix(
    x, y, num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    p_mix: float,
):
    y1 = one_hot(y, num_classes)

    if p_mix <= 0 or torch.rand(1).item() > p_mix:
        return x, y1, False

    use_cutmix = (cutmix_alpha > 0) and (torch.rand(1).item() > 0.5)

    if not use_cutmix:
        if mixup_alpha <= 0:
            return x, y1, False
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        perm = torch.randperm(x.size(0), device=x.device)
        x2 = x[perm]
        y2 = y1[perm]
        x_aug = lam * x + (1 - lam) * x2
        y_soft = lam * y1 + (1 - lam) * y2
        return x_aug, y_soft, True

    lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = x[perm]
    y2 = y1[perm]

    _, _, H, W = x.shape
    x1b, y1b, x2b, y2b = rand_bbox(W, H, lam)

    x_aug = x.clone()
    x_aug[:, :, y1b:y2b, x1b:x2b] = x2[:, :, y1b:y2b, x1b:x2b]

    area = (x2b - x1b) * (y2b - y1b)
    lam_adj = 1.0 - (area / (W * H + 1e-12))
    y_soft = lam_adj * y1 + (1 - lam_adj) * y2
    return x_aug, y_soft, True


def soft_cross_entropy_weighted(
    logits: torch.Tensor,
    y_soft: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    if class_weights is None:
        return -(y_soft * logp).sum(dim=1).mean()

    w = class_weights.view(1, -1).to(logits.device)
    num = -(y_soft * logp * w).sum(dim=1)
    den = (y_soft * w).sum(dim=1).clamp_min(1e-12)
    return (num / den).mean()


# -----------------------------
# Fairness helpers
# -----------------------------
def compute_class_weights_effective_num(
    y: List[int],
    num_classes: int,
    beta: float = 0.9999
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for yi in y:
        if 0 <= yi < num_classes:
            counts[yi] += 1.0
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights


def compute_class_priors(y: List[int], num_classes: int, eps: float = 1e-12) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for yi in y:
        if 0 <= yi < num_classes:
            counts[yi] += 1.0
    priors = counts + eps
    priors = priors / priors.sum().clamp_min(eps)
    return priors


def logit_adjustment(logits: torch.Tensor, class_priors: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    return logits + tau * torch.log(class_priors.to(logits.device).clamp_min(1e-12))


def tau_schedule(epoch: int, total_epochs: int, tau_start: float, tau_end: float, warmup: int) -> float:
    """
    Start strong to break majority-class collapse, then reduce to improve precision/stability.
    """
    if total_epochs <= 1:
        return float(tau_end)
    if epoch <= warmup:
        return float(tau_start)
    t = (epoch - warmup) / float(max(1, total_epochs - warmup))
    t = min(max(t, 0.0), 1.0)
    return float(tau_start + (tau_end - tau_start) * t)


# -----------------------------
# Optional: Focal Loss (hard labels only)
# -----------------------------
class FocalCE(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 0.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight_buf", weight if weight is not None else torch.tensor([]))
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.weight_buf.numel() > 0:
            ce = F.cross_entropy(
                logits, target,
                weight=self.weight_buf.to(logits.device),
                label_smoothing=max(0.0, self.label_smoothing),
                reduction="none"
            )
        else:
            ce = F.cross_entropy(
                logits, target,
                label_smoothing=max(0.0, self.label_smoothing),
                reduction="none"
            )
        if self.gamma <= 0.0:
            return ce.mean()

        # p_t = exp(-CE)
        pt = torch.exp(-ce).clamp_min(1e-12)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


# -----------------------------
# Scheduler (warmup + cosine)
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
# Transforms
# -----------------------------
def build_transforms(img_size: int, mean, std, train: bool, random_erasing: float = 0.0):
    if train:
        t = [
            transforms.RandomResizedCrop(img_size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.015),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        if random_erasing and random_erasing > 0:
            t.append(transforms.RandomErasing(p=float(random_erasing), scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"))
        return transforms.Compose(t)
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
class Metrics:
    loss: float
    acc: float
    bal_acc: float
    macro_f1: float
    macro_prec: float
    macro_rec: float


def metrics_from_confmat(conf: torch.Tensor) -> Tuple[float, float, float, float, float]:
    conf_f = conf.float()
    tp = torch.diag(conf_f)
    support = conf_f.sum(dim=1)
    total = support.sum().clamp_min(1.0)

    acc = (tp.sum() / total).item()

    recall = tp / (support.clamp_min(1e-12))
    bal_acc = recall.mean().item()

    fp = conf_f.sum(dim=0) - tp
    precision = tp / (tp + fp + 1e-12)

    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    macro_f1 = f1.mean().item()
    macro_prec = precision.mean().item()
    macro_rec = recall.mean().item()
    return acc, bal_acc, macro_f1, macro_prec, macro_rec


def print_per_class(conf: torch.Tensor, classes: List[str], title: str):
    conf = conf.detach().cpu().float()
    tp = torch.diag(conf)
    support = conf.sum(dim=1).clamp_min(1.0)
    pred_sum = conf.sum(dim=0).clamp_min(1.0)

    recall = (tp / support)
    precision = (tp / pred_sum)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    order = torch.argsort(support, descending=True)
    print(title)
    print(f"{'class':<12} {'n':>6} {'rec':>8} {'prec':>8} {'f1':>8}")
    for i in order.tolist():
        n_i = int(support[i].item())
        print(f"{classes[i]:<12} {n_i:>6d} {recall[i].item()*100:>7.2f}% {precision[i].item()*100:>7.2f}% {f1[i].item():>8.4f}")


# -----------------------------
# TTA (simple)
# -----------------------------
@torch.no_grad()
def forward_tta_flip(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits1 = model(x)
    logits2 = model(torch.flip(x, dims=[3]))  # horizontal flip
    return (logits1 + logits2) / 2.0


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
    num_classes: int,
    label_smoothing: float,
    mix_p: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    grad_clip: float,
    accum_steps: int,
    class_weights: Optional[torch.Tensor],
    class_priors: Optional[torch.Tensor],
    tau_now: float,
    focal_gamma: float,
) -> float:
    model.train()

    focal = FocalCE(weight=(class_weights if class_weights is not None else None),
                    gamma=float(focal_gamma),
                    label_smoothing=max(0.0, float(label_smoothing)))

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

        x_aug, y_soft, did_mix = apply_mixup_cutmix(
            x, y, num_classes=num_classes,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, p_mix=mix_p
        )

        with autocast_context(autocast_fn, device, enabled=use_amp):
            logits = model(x_aug)

            if (class_priors is not None) and (tau_now != 0.0):
                logits = logit_adjustment(logits, class_priors, tau=tau_now)

            if did_mix:
                loss = soft_cross_entropy_weighted(logits, y_soft, class_weights=class_weights)
            else:
                loss = focal(logits, y)

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

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            batch_acc = (pred == y).float().mean().item()

        avg_loss = running / max(1, n)
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{batch_acc*100:.2f}%", lr=f"{lr_now:.2e}", upd=updates, accum=accum_steps, tau=f"{tau_now:.2f}")

    return running / max(1, n)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_weights: Optional[torch.Tensor],
    class_priors: Optional[torch.Tensor],
    tau_now: float,
    tta: bool,
    desc: str = "val",
) -> Tuple[Metrics, torch.Tensor]:
    model.eval()

    if class_weights is not None:
        ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    n = 0
    conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        logits = forward_tta_flip(model, x) if tta else model(x)

        if (class_priors is not None) and (tau_now != 0.0):
            logits = logit_adjustment(logits, class_priors, tau=tau_now)

        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        pred = logits.argmax(dim=1)
        idx = (y * num_classes + pred).to(torch.int64)
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        conf += binc.view(num_classes, num_classes)

        acc, bal_acc, macro_f1, macro_prec, macro_rec = metrics_from_confmat(conf)
        avg_loss = total_loss / max(1, n)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc*100:.2f}%", bal=f"{bal_acc*100:.2f}%", f1=f"{macro_f1:.3f}", tau=f"{tau_now:.2f}")

    avg_loss = total_loss / max(1, n)
    acc, bal_acc, macro_f1, macro_prec, macro_rec = metrics_from_confmat(conf)
    return Metrics(loss=avg_loss, acc=acc, bal_acc=bal_acc, macro_f1=macro_f1, macro_prec=macro_prec, macro_rec=macro_rec), conf


# -----------------------------
# Param groups: differential LR
# -----------------------------
def build_optimizer(model: nn.Module, lr: float, wd: float, head_lr_mult: float = 3.0):
    """
    Backbone gets lr, head gets lr * head_lr_mult.
    Helps stability on noisy labels, and helps minority head adapt.
    """
    head_params = []
    back_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(p)
        else:
            back_params.append(p)

    param_groups = [
        {"params": back_params, "lr": lr, "weight_decay": wd},
        {"params": head_params, "lr": lr * head_lr_mult, "weight_decay": wd},
    ]
    return torch.optim.AdamW(param_groups)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default=None)
    ap.add_argument("--root_dir", default=None)
    ap.add_argument("--img_col", default=None)
    ap.add_argument("--label_col", default=None)

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--warmup_epochs", type=int, default=2)

    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--img_size_ft", type=int, default=512)
    ap.add_argument("--ft_from_epoch", type=int, default=11)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_ft", type=float, default=8e-5)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--head_lr_mult", type=float, default=3.0)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--effective_batch", type=int, default=64)
    ap.add_argument("--accum_steps", type=int, default=None)

    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--out", default="polish_best.pt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_decode_retries", type=int, default=10)

    ap.add_argument("--crop_cache", action="store_true", default=True)
    ap.add_argument("--no_crop_cache", action="store_true")
    ap.add_argument("--cache_dir", default="_polish_cache_crops")

    ap.add_argument("--sampler", action="store_true", default=True)
    ap.add_argument("--no_sampler", action="store_true")

    ap.add_argument("--label_smoothing", type=float, default=0.05)

    ap.add_argument("--mix_p", type=float, default=0.6)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--cutmix_alpha", type=float, default=0.5)

    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--per_class", action="store_true", default=True)
    ap.add_argument("--no_per_class", action="store_true")

    # Fairness knobs
    ap.add_argument("--cb_beta", type=float, default=0.9999)
    ap.add_argument("--tau_start", type=float, default=0.9, help="initial logit adjustment strength")
    ap.add_argument("--tau_end", type=float, default=0.4, help="final logit adjustment strength")
    ap.add_argument("--tau_warmup", type=int, default=3, help="epochs to keep tau_start before decaying")
    ap.add_argument("--focal_gamma", type=float, default=1.0, help="0 disables focal; >0 helps hard examples (no mix)")

    # Aug knobs
    ap.add_argument("--random_erasing", type=float, default=0.15, help="RandomErasing prob in train")
    ap.add_argument("--tta", action="store_true", default=False, help="TTA flip in validation")

    args = ap.parse_args()

    if args.no_crop_cache:
        args.crop_cache = False
    if args.no_sampler:
        args.sampler = False
    if args.no_ema:
        args.ema = False
    if args.no_per_class:
        args.per_class = False

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
    df[label_col] = df[label_col].astype(str).str.strip()

    # Build cache crops (optional)
    if args.crop_cache:
        cache_root = os.path.join(os.path.dirname(args.csv), args.cache_dir)
        print(f"[INFO] Crop cache: ON -> {cache_root}")
        df = preprocess_crops(df, img_col=img_col, root_dir=args.root_dir, cache_dir=cache_root)
        args.root_dir = cache_root
    else:
        print("[INFO] Crop cache: OFF")

    # Classes
    classes = sorted(df[label_col].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    num_classes = len(classes)

    print(f"[INFO] img_col={img_col} label_col={label_col} num_classes={num_classes}")
    print(f"[INFO] classes={classes}")
    print(f"[INFO] GradAccum: batch={args.batch} effective_batch={args.effective_batch} accum_steps={args.accum_steps}")

    # Split
    train_idx, val_idx = stratified_split_indices(df[label_col].tolist(), args.val_split, args.seed)
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # Image index
    image_index = build_image_index(args.root_dir)

    # Transforms
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std
    tfm_train = build_transforms(args.img_size, mean, std, train=True, random_erasing=args.random_erasing)
    tfm_val = build_transforms(args.img_size_ft, mean, std, train=False, random_erasing=0.0)

    ds_train = PolishDataset(
        df_train, img_col, label_col, args.root_dir, label2idx, image_index,
        tfm=tfm_train, max_decode_retries=args.max_decode_retries
    )
    ds_val = PolishDataset(
        df_val, img_col, label_col, args.root_dir, label2idx, image_index,
        tfm=tfm_val, max_decode_retries=args.max_decode_retries
    )

    # Sampler
    sampler = None
    if args.sampler:
        y_train_str = df_train[label_col].tolist()
        counts: Dict[str, int] = {}
        for y in y_train_str:
            counts[y] = counts.get(y, 0) + 1
        sample_weights = [1.0 / counts[y] for y in y_train_str]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print("[INFO] Sampler: WeightedRandomSampler ON")
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

    # Device + model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = PolishNet(num_classes=num_classes, dropout=args.dropout).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Differential LR optimizer
    optimizer = build_optimizer(model, lr=args.lr, wd=args.wd, head_lr_mult=args.head_lr_mult)

    # AMP
    autocast_fn, scaler, amp_available = get_amp(device)
    use_amp = amp_available and (not args.no_amp)

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # Scheduler
    updates_per_epoch = int(math.ceil(len(dl_train) / max(1, args.accum_steps)))
    warmup_steps = updates_per_epoch * max(0, int(args.warmup_epochs))
    total_steps = updates_per_epoch * max(1, int(args.epochs))
    sched = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    sched.set_lr(0)

    # Fairness stats from TRAIN labels
    y_train_int = [label2idx[str(v).strip()] for v in df_train[label_col].tolist()]
    class_weights = compute_class_weights_effective_num(y_train_int, num_classes=num_classes, beta=args.cb_beta)
    class_priors = compute_class_priors(y_train_int, num_classes=num_classes)

    print(f"[INFO] Class weights (effective-num, beta={args.cb_beta}): {class_weights.tolist()}")
    print(f"[INFO] Class priors: {class_priors.tolist()}")
    print(f"[INFO] Tau schedule: start={args.tau_start} end={args.tau_end} warmup={args.tau_warmup}")
    print(f"[INFO] Focal gamma={args.focal_gamma} (applies only when no MixUp/CutMix in that batch)")
    print(f"[INFO] head_lr_mult={args.head_lr_mult} | random_erasing={args.random_erasing} | TTA={args.tta}")

    best_bal = -1.0

    for epoch in range(1, args.epochs + 1):
        tau_now = tau_schedule(epoch, args.epochs, args.tau_start, args.tau_end, warmup=args.tau_warmup)

        # Fine-tune stage
        if epoch == args.ft_from_epoch:
            print(f"[STAGE] Fine-tune from epoch {epoch}: img_size={args.img_size_ft}, lr_ft={args.lr_ft}")
            ds_train.tfm = build_transforms(args.img_size_ft, mean, std, train=True, random_erasing=args.random_erasing)
            for pg_i, pg in enumerate(optimizer.param_groups):
                # keep head multiplier even in ft stage
                base = args.lr_ft
                if pg_i == 1:
                    pg["lr"] = base * args.head_lr_mult
                else:
                    pg["lr"] = base

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
            num_classes=num_classes,
            label_smoothing=args.label_smoothing,
            mix_p=args.mix_p if epoch < args.ft_from_epoch else 0.0,
            mixup_alpha=args.mixup_alpha if epoch < args.ft_from_epoch else 0.0,
            cutmix_alpha=args.cutmix_alpha if epoch < args.ft_from_epoch else 0.0,
            grad_clip=args.grad_clip,
            accum_steps=args.accum_steps,
            class_weights=class_weights,
            class_priors=class_priors,
            tau_now=tau_now,
            focal_gamma=args.focal_gamma if epoch < args.ft_from_epoch else 0.0,  # focal mostly useful pre-ft
        )

        # Validate with EMA weights if enabled
        if ema is not None:
            ema.apply_shadow(model)

        val_metrics, conf = evaluate(
            model, dl_val, device, num_classes=num_classes,
            class_weights=class_weights,
            class_priors=class_priors,
            tau_now=tau_now,
            tta=bool(args.tta),
            desc="val"
        )

        if ema is not None:
            ema.restore(model)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"tau={tau_now:.2f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_acc={val_metrics.acc*100:.2f}% | "
            f"val_bal_acc={val_metrics.bal_acc*100:.2f}% | "
            f"val_macroF1={val_metrics.macro_f1:.4f} | "
            f"val_macroP={val_metrics.macro_prec*100:.2f}% | "
            f"val_macroR={val_metrics.macro_rec*100:.2f}%"
        )

        if args.per_class:
            print_per_class(conf, classes, title=f"[VAL] Per-class @ epoch {epoch:02d}")

        if val_metrics.bal_acc > best_bal:
            best_bal = val_metrics.bal_acc

            # save model + optionally EMA shadow
            ckpt = {
                "model": model.state_dict(),
                "ema": (ema.shadow if ema is not None else None),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "classes": classes,
                "img_col": img_col,
                "label_col": label_col,
                "root_dir": args.root_dir,
                "backbone": "convnext_tiny",
                "weights": "ConvNeXt_Tiny_Weights.DEFAULT",
                "task": "polish_classification",
                "fairness": {
                    "class_weights_effective_num": class_weights.tolist(),
                    "class_priors": class_priors.tolist(),
                    "cb_beta": args.cb_beta,
                    "tau_start": args.tau_start,
                    "tau_end": args.tau_end,
                    "tau_warmup": args.tau_warmup,
                },
                "train": {
                    "img_size": args.img_size,
                    "img_size_ft": args.img_size_ft,
                    "ft_from_epoch": args.ft_from_epoch,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "lr_ft": args.lr_ft,
                    "wd": args.wd,
                    "head_lr_mult": args.head_lr_mult,
                    "batch": args.batch,
                    "effective_batch": args.effective_batch,
                    "accum_steps": args.accum_steps,
                    "label_smoothing": args.label_smoothing,
                    "mix": {"p": args.mix_p, "mixup_alpha": args.mixup_alpha, "cutmix_alpha": args.cutmix_alpha},
                    "focal_gamma": args.focal_gamma,
                    "random_erasing": args.random_erasing,
                    "tta": bool(args.tta),
                    "ema_decay": (args.ema_decay if args.ema else None),
                }
            }
            torch.save(ckpt, args.out)
            print(f"[SAVE] best -> {args.out} (val_bal_acc={best_bal*100:.2f}%)")

    print(f"[DONE] best_val_bal_acc={best_bal*100:.2f}% | saved={args.out}")


if __name__ == "__main__":
    main()
