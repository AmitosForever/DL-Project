import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

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
# Utils
# -----------------------------
def seed_all(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Speed freebies on Ampere+ (safe for CV training)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def guess_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols = [c.lower() for c in columns]
    for cand in candidates:
        for i, c in enumerate(cols):
            if cand in c:
                return columns[i]
    return None


def normalize_clarity_label(s: str, merge_si3_to_si2: bool = True) -> str:
    """
    Clean noisy labels:
      - SI2- -> SI2
      - I4.. -> I3 (standard is I1-I3)
      - optional: SI3 -> SI2
    """
    if s is None:
        return ""
    s = str(s).strip().upper()

    if s == "SI2-":
        s = "SI2"
    if merge_si3_to_si2 and s == "SI3":
        s = "SI2"

    if s.startswith("I") and len(s) == 2 and s[1].isdigit():
        try:
            v = int(s[1])
            if v >= 4:
                s = "I3"
        except Exception:
            pass
    return s


def stratified_split_indices(labels: List[str], val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    import random
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


def resolve_path(p: str, root_dir: Optional[str]) -> str:
    p = str(p).replace("/", os.sep)
    if root_dir and not os.path.isabs(p):
        return os.path.join(root_dir, p)
    return p


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
            df = pd.read_csv(p, nrows=80)
        except Exception:
            return -999
        cols = [c.lower() for c in df.columns.tolist()]
        s = 0
        if any("clarity" in c for c in cols):
            s += 10
        if any(k in c for c in cols for k in ["image", "img", "path", "file", "filename"]):
            s += 5
        return s

    scored = [(score_csv(p), p) for p in csvs]
    scored.sort(reverse=True, key=lambda x: x[0])
    best_score, best_path = scored[0]
    return best_path if best_score >= 0 else csvs[0]


# -----------------------------
# Crop cache (diamond-focused)
# -----------------------------
def crop_diamond_pil(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def crop_diamond_cv2(img: Image.Image) -> Image.Image:
    import cv2
    import numpy as np

    rgb = np.array(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_diamond_pil(img)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    H, W = gray.shape[:2]
    if w * h < 0.02 * (W * H):
        return crop_diamond_pil(img)

    pad = int(0.08 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    crop = img.crop((x0, y0, x1, y1))
    return crop_diamond_pil(crop)


def get_cropped_path(cache_dir: str, rel_path: str) -> str:
    rel_path = rel_path.replace("/", os.sep)
    base, _ = os.path.splitext(rel_path)
    out_path = os.path.join(cache_dir, base + ".jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def preprocess_crops(
    df: pd.DataFrame,
    img_col: str,
    root_dir: Optional[str],
    cache_dir: str,
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    """
    UPGRADE:
    - If crop fails, store ABSOLUTE original path so that later root_dir switch to cache_root won't break those rows.
    """
    os.makedirs(cache_dir, exist_ok=True)

    has_cv2 = False
    try:
        import cv2  # noqa
        has_cv2 = True
    except Exception:
        has_cv2 = False

    new_paths = []
    n = len(df) if max_items is None else min(len(df), max_items)

    pbar = tqdm(range(n), desc="preprocess_crops", leave=True)
    bad = 0
    for i in pbar:
        rel = str(df.iloc[i][img_col])
        src = resolve_path(rel, root_dir)
        dst = get_cropped_path(cache_dir, rel)

        if os.path.exists(dst):
            new_paths.append(os.path.relpath(dst, cache_dir))
            continue

        try:
            img = Image.open(src).convert("RGB")
            img = crop_diamond_cv2(img) if has_cv2 else crop_diamond_pil(img)
            img.save(dst, quality=92, optimize=True)
            new_paths.append(os.path.relpath(dst, cache_dir))
        except Exception:
            bad += 1
            # IMPORTANT UPGRADE: keep absolute path to the original file (if possible)
            new_paths.append(src)

        if (i + 1) % 500 == 0:
            pbar.set_postfix(bad=bad)

    df2 = df.copy()
    df2[img_col] = new_paths + df2[img_col].iloc[len(new_paths):].tolist()
    return df2


# -----------------------------
# Dataset (robust)
# -----------------------------
class ClarityDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_col: str,
        label_col: str,
        root_dir: Optional[str],
        label2idx: Dict[str, int],
        tfm=None,
        max_decode_retries: int = 10,
        class_ranks: Optional[torch.Tensor] = None,   # [C] float ranks (CPU)
        return_rank: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.root_dir = root_dir
        self.label2idx = label2idx
        self.tfm = tfm
        self.max_decode_retries = max_decode_retries
        self.class_ranks = class_ranks
        self.return_rank = return_rank

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

            img_path = resolve_path(p, self.root_dir)
            try:
                img = Image.open(img_path).convert("RGB")
                if self.tfm is not None:
                    img = self.tfm(img)
                y_idx = self.label2idx[y_str]

                if self.return_rank:
                    if self.class_ranks is None:
                        raise RuntimeError("return_rank=True but class_ranks is None")
                    y_rank = float(self.class_ranks[y_idx].item())
                    return img, y_idx, torch.tensor(y_rank, dtype=torch.float32)

                return img, y_idx
            except Exception as e:
                last_err = (img_path, repr(e))
                idx = torch.randint(0, len(self.df), (1,)).item()

        raise RuntimeError(f"Decode failed. Last: {last_err[0]} | {last_err[1]}")


# -----------------------------
# Model (ConvNeXt-Tiny)
# -----------------------------
class ClarityNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1, task: str = "ordinal_mse"):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)

        in_features = self.backbone.classifier[2].in_features  # 768
        out_dim = 1 if task == "ordinal_mse" else num_classes

        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, out_dim),
        )

        self.task = task

        # -----------------------------
        # UPDATED: stability init for ordinal regression (approx midpoint ~5.0 on 0..11.5 scale)
        # -----------------------------
        if self.task == "ordinal_mse":
            try:
                nn.init.constant_(self.backbone.classifier[3].bias, 5.0)
            except Exception:
                final = self.backbone.classifier[-1]
                if isinstance(final, nn.Linear) and final.bias is not None:
                    nn.init.constant_(final.bias, 5.0)

    def forward(self, x):
        out = self.backbone(x)
        if self.task == "ordinal_mse":
            return out.squeeze(-1)  # [B]
        return out  # [B,C]


# -----------------------------
# EMA
# -----------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
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
# AMP (UPGRADED: safe across torch versions)
# -----------------------------
def get_amp(device: torch.device):
    """
    Returns:
      autocast_fn, scaler, use_amp_possible
    """
    if device.type != "cuda":
        return None, None, False

    # Prefer torch.amp.* (new API) if exists
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.amp.GradScaler(enabled=True)
        return torch.amp.autocast, scaler, True

    # Fallback old cuda.amp
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    return torch.cuda.amp.autocast, scaler, True


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


def autocast_context(autocast_fn, device: torch.device, enabled: bool):
    """
    Works with both:
      - torch.amp.autocast(device_type, enabled=...)
      - torch.cuda.amp.autocast(enabled=...)
    """
    if (not enabled) or (autocast_fn is None) or (device.type != "cuda"):
        return _NullCtx()

    # Try new-style signature first
    try:
        return autocast_fn("cuda", enabled=True)
    except TypeError:
        return autocast_fn(enabled=True)


# -----------------------------
# MixUp/CutMix + soft CE
# -----------------------------
def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=num_classes).float()


def smooth_one_hot(y_hard: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return one_hot(y_hard, num_classes)
    y = one_hot(y_hard, num_classes)
    return y * (1.0 - smoothing) + (smoothing / float(num_classes))


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


def soft_cross_entropy(logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(y_soft * logp).sum(dim=1).mean()


# -----------------------------
# MixUp for scalar regression (ordinal MSE)
# -----------------------------
def apply_mixup_regression(
    x: torch.Tensor,
    y_rank: torch.Tensor,
    mixup_alpha: float,
    p_mix: float,
):
    """
    MixUp for scalar targets (ordinal regression).
    y_rank: [B] float
    """
    if p_mix <= 0 or mixup_alpha <= 0 or torch.rand(1).item() > p_mix:
        return x, y_rank, False

    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = x[perm]
    y2 = y_rank[perm]
    x_aug = lam * x + (1 - lam) * x2
    y_aug = lam * y_rank + (1 - lam) * y2
    return x_aug, y_aug, True


# -----------------------------
# Focal MSE Loss (Version 8)
# -----------------------------
class FocalMSELoss(nn.Module):
    """
    Focal MSE for regression:
      mse = (pred - target)^2
      w = (1 - exp(-mse))^gamma     # down-weights easy (small mse), saturates for hard
      loss = w * mse

    Supports reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, gamma: float = 0.5, reduction: str = "mean", eps: float = 1e-12):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = str(reduction)
        self.eps = float(eps)

        if self.reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        mse = (pred - target) ** 2  # [B]
        w = (1.0 - torch.exp(-mse)).clamp_min(self.eps) ** self.gamma  # [B]
        loss = w * mse  # [B]

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


# -----------------------------
# Clarity-specific: Ordinal regularization (kept for compatibility)
# -----------------------------
def build_clarity_rank(classes: List[str]) -> torch.Tensor:
    # UPDATED: Mildly adjusted ordinal scale.
    # Linear gaps (1.0) for most classes, but a gentle 1.5 gap before I1.
    rank_map = {
        "FL": 0.0, "IF": 1.0,
        "VVS1": 2.0, "VVS2": 3.0,
        "VS1": 4.0, "VS2": 5.0,
        "SI1": 6.0, "SI2": 7.0,
        "I1": 8.5,   # Gap 1.5 (gentle push)
        "I2": 10.0,
        "I3": 11.5
    }

    ranks = []
    for c in classes:
        if c in rank_map:
            ranks.append(float(rank_map[c]))
        else:
            # Keep deterministic fallback for any unexpected labels
            ranks.append(100.0 + float(sorted(classes).index(c)))
    return torch.tensor(ranks, dtype=torch.float32)  # [C]


def ordinal_expected_rank_loss(logits: torch.Tensor, y: torch.Tensor, class_ranks: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    exp_rank = (probs * class_ranks.unsqueeze(0)).sum(dim=1)  # [B]
    true_rank = class_ranks[y]  # [B]
    return F.mse_loss(exp_rank, true_rank)


# -----------------------------
# Metrics + reporting
# -----------------------------
@dataclass
class Metrics:
    loss: float
    acc: float
    macro_f1: float


def metrics_from_confmat(conf: torch.Tensor) -> Tuple[float, float]:
    conf_f = conf.float()
    tp = torch.diag(conf_f)
    support = conf_f.sum(dim=1)
    total = support.sum().clamp_min(1.0)
    acc = tp.sum() / total

    fp = conf_f.sum(dim=0) - tp
    fn = support - tp
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    valid = support > 0
    macro_f1 = f1[valid].mean() if valid.any() else f1.mean()
    return acc.item(), macro_f1.item()


def print_per_class_report(conf: torch.Tensor, classes: List[str], title: str = "[VAL] Per-class"):
    conf = conf.detach().cpu().to(torch.float32)
    tp = torch.diag(conf)
    support = conf.sum(dim=1).clamp_min(1.0)
    pred_sum = conf.sum(dim=0).clamp_min(1.0)

    acc_c = (tp / support)
    precision = tp / pred_sum
    recall = tp / support
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    supports_int = conf.sum(dim=1).to(torch.int64)
    order = torch.argsort(supports_int, descending=True)

    print(title)
    print(f"{'class':<6} {'n':>6} {'acc':>8} {'prec':>8} {'rec':>8} {'f1':>8}")
    for i in order.tolist():
        n_i = int(supports_int[i].item())
        if n_i == 0:
            continue
        print(
            f"{classes[i]:<6} {n_i:>6d} "
            f"{acc_c[i].item()*100:>7.2f}% "
            f"{precision[i].item()*100:>7.2f}% "
            f"{recall[i].item()*100:>7.2f}% "
            f"{f1[i].item():>8.4f}"
        )


def print_confusion_matrix(conf: torch.Tensor, classes: List[str], normalize: bool, title: str):
    conf_cpu = conf.detach().cpu()
    if normalize:
        row_sum = conf_cpu.sum(dim=1, keepdim=True).clamp_min(1)
        mat = (conf_cpu.float() / row_sum.float()).numpy()
    else:
        mat = conf_cpu.numpy()

    df = pd.DataFrame(mat, index=classes, columns=classes)
    print(title)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220):
        if normalize:
            print(df.to_string(float_format=lambda x: f"{x:.2f}"))
        else:
            print(df.to_string())


def print_top_confusions(conf: torch.Tensor, classes: List[str], k: int = 2, title: str = "[VAL] Top confusions"):
    conf_cpu = conf.detach().cpu().to(torch.int64)
    C = conf_cpu.shape[0]
    print(title)
    for i in range(C):
        row = conf_cpu[i].clone()
        total = int(row.sum().item())
        if total == 0:
            continue
        row[i] = 0
        vals, idxs = torch.topk(row, k=min(k, C))
        pairs = []
        for v, j in zip(vals.tolist(), idxs.tolist()):
            if v <= 0:
                continue
            pairs.append(f"{classes[j]}:{v}")
        pairs_str = ", ".join(pairs) if pairs else "-"
        print(f"  {classes[i]:<6} (n={total:>4d}) -> {pairs_str}")


# -----------------------------
# TTA (real five-crop): corners+center with smaller crop
# -----------------------------
@torch.no_grad()
def tta_logits(model: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    mode: "none", "flip", "fivecrop", "fivecrop_flip"
    Returns:
      - classification: [B,C]
      - regression: [B]
    """
    if mode == "none":
        return model(x)

    if mode == "flip":
        p1 = model(x)
        p2 = model(torch.flip(x, dims=[3]))
        return (p1 + p2) / 2.0

    B, C, H, W = x.shape
    s = min(H, W)
    crop = int(round(s * 0.875))  # e.g., 512 -> 448
    crop = max(32, min(crop, s))

    tl = x[..., 0:crop, 0:crop]
    tr = x[..., 0:crop, W - crop:W]
    bl = x[..., H - crop:H, 0:crop]
    br = x[..., H - crop:H, W - crop:W]
    hs = (H - crop) // 2
    ws = (W - crop) // 2
    cc = x[..., hs:hs + crop, ws:ws + crop]

    crops = [tl, tr, bl, br, cc]

    out_sum = None
    n = 0
    for cimg in crops:
        o = model(cimg)
        out_sum = o if out_sum is None else (out_sum + o)
        n += 1
        if mode == "fivecrop_flip":
            of = model(torch.flip(cimg, dims=[3]))
            out_sum = out_sum + of
            n += 1

    return out_sum / float(n)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion_hard,
    tta_mode: str = "none",
    lr: Optional[float] = None,
    show_pbar: bool = True,
    desc: str = "val",
    task: str = "ordinal_mse",
    class_ranks: Optional[torch.Tensor] = None,  # [C] float (device)
    classes: Optional[List[str]] = None,
) -> Tuple[Metrics, torch.Tensor]:
    """
    Validation:
      - ordinal_mse: model outputs [B] predicted rank; prediction -> nearest class rank (argmin distance).
      - classification: model outputs logits [B,C].
    """
    model.eval()
    total_loss = 0.0
    n = 0

    if classes is None:
        raise RuntimeError("evaluate(): classes must be provided")
    num_classes = len(classes)
    conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    it = loader
    pbar = None
    if show_pbar:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        it = pbar

    for batch in it:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y_idx, y_rank = batch
        else:
            x, y_idx = batch
            y_rank = None

        x = x.to(device, non_blocking=True)
        y_idx = y_idx.to(device, non_blocking=True)

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        if task == "ordinal_mse":
            if class_ranks is None:
                raise RuntimeError("evaluate(): class_ranks required for ordinal_mse")
            if y_rank is None:
                y_rank = class_ranks[y_idx].to(device)
            else:
                y_rank = y_rank.to(device, non_blocking=True).view(-1)

            pred_rank = tta_logits(model, x, mode=tta_mode).view(-1)  # [B]
            loss = criterion_hard(pred_rank, y_rank)

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

            # Map predicted scalar -> nearest class by rank distance (supports non-integer ranks like 8.5)
            d = torch.abs(pred_rank.unsqueeze(1) - class_ranks.unsqueeze(0))  # [B,C]
            pred_idx = d.argmin(dim=1).to(torch.long)

            idx = (y_idx * num_classes + pred_idx).to(torch.int64)
            binc = torch.bincount(idx, minlength=num_classes * num_classes)
            conf += binc.view(num_classes, num_classes)

        else:
            logits = tta_logits(model, x, mode=tta_mode)
            loss = criterion_hard(logits, y_idx)

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

            pred = logits.argmax(dim=1)
            idx = (y_idx * num_classes + pred).to(torch.int64)
            binc = torch.bincount(idx, minlength=num_classes * num_classes)
            conf += binc.view(num_classes, num_classes)

        if pbar is not None:
            avg_loss = total_loss / max(n, 1)
            acc_now, _ = metrics_from_confmat(conf)
            lr_str = f"{lr:.2e}" if lr is not None else "-"
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc_now*100:.2f}%", lr=lr_str)

    avg_loss = total_loss / max(n, 1)
    acc, macro_f1 = metrics_from_confmat(conf)
    return Metrics(loss=avg_loss, acc=acc, macro_f1=macro_f1), conf


# -----------------------------
# Manual Warmup+Cosine scheduler (no PyTorch warning nonsense)
# -----------------------------
class WarmupCosine:
    """
    Steps are "per-optimizer-update" (not per micro-batch when using grad accumulation).
    Usage:
      sched = WarmupCosine(optimizer, warmup_steps, total_steps)
      sched.set_lr(0)  # set initial LR for step=0 BEFORE training
      ...
      optimizer.step()
      sched.step()     # advances to next step and sets LR
    """
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
# Train loop (with Gradient Accumulation + ordinal MSE)
# -----------------------------
def train_one_epoch(
    model,
    loader,
    device,
    optimizer,
    autocast_fn,
    scaler,
    use_amp: bool,
    num_classes: int,
    mix_p: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    label_smoothing: float,
    log_every: int,
    scheduler: WarmupCosine,
    ema: Optional[ModelEMA],
    grad_clip: float,
    ordinal_lambda: float,
    class_ranks: Optional[torch.Tensor],

    # NEW
    task: str = "ordinal_mse",
    accum_steps: int = 1,
    class_weights: Optional[torch.Tensor] = None,  # [C] float weights for imbalance (optional)
    focal_gamma: float = 0.5,                      # Version 8: focal regression gamma
):
    model.train()
    running = 0.0
    n = 0
    t0 = time.time()

    accum_steps = max(1, int(accum_steps))

    hard_criterion = nn.CrossEntropyLoss(label_smoothing=max(0.0, float(label_smoothing)))
    focal_mse_none = FocalMSELoss(gamma=focal_gamma, reduction="none")

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc="train", leave=False, dynamic_ncols=True)

    micro_step = 0
    update_step = 0

    for step, batch in pbar:
        micro_step += 1

        # Support both (x,y) and (x,y,y_rank)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y_idx, y_rank = batch
        else:
            x, y_idx = batch
            y_rank = None

        x = x.to(device, non_blocking=True)
        y_idx = y_idx.to(device, non_blocking=True)

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        did_mix = False

        if task == "ordinal_mse":
            if class_ranks is None:
                raise RuntimeError("train_one_epoch(): class_ranks required for ordinal_mse")

            if y_rank is None:
                y_rank = class_ranks[y_idx]
            else:
                y_rank = y_rank.to(device, non_blocking=True).view(-1)

            x_aug, y_aug, did_mix = apply_mixup_regression(
                x, y_rank, mixup_alpha=mixup_alpha, p_mix=mix_p
            )

            if use_amp:
                with autocast_context(autocast_fn, device, enabled=True):
                    pred_rank = model(x_aug).view(-1)  # [B]
                    loss_vec = focal_mse_none(pred_rank, y_aug)  # [B]

                    if class_weights is not None:
                        w = class_weights[y_idx].to(device)
                        loss_vec = loss_vec * w

                    loss = loss_vec.mean()

                (scaler.scale(loss) / accum_steps).backward()
            else:
                pred_rank = model(x_aug).view(-1)
                loss_vec = focal_mse_none(pred_rank, y_aug)
                if class_weights is not None:
                    w = class_weights[y_idx].to(device)
                    loss_vec = loss_vec * w
                loss = loss_vec.mean()
                (loss / accum_steps).backward()

        else:
            # Original classification path (kept)
            x_aug, y_soft, did_mix = apply_mixup_cutmix(
                x, y_idx, num_classes=num_classes,
                mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, p_mix=mix_p
            )

            if use_amp:
                with autocast_context(autocast_fn, device, enabled=True):
                    logits = model(x_aug)
                    if did_mix:
                        loss_main = soft_cross_entropy(logits, y_soft)
                    else:
                        loss_main = hard_criterion(logits, y_idx)

                    loss = loss_main
                    if ordinal_lambda > 0 and class_ranks is not None:
                        loss = loss + (ordinal_lambda * ordinal_expected_rank_loss(logits, y_idx, class_ranks))

                (scaler.scale(loss) / accum_steps).backward()
            else:
                logits = model(x_aug)
                if did_mix:
                    loss_main = soft_cross_entropy(logits, y_soft)
                else:
                    loss_main = hard_criterion(logits, y_idx)

                loss = loss_main
                if ordinal_lambda > 0 and class_ranks is not None:
                    loss = loss + (ordinal_lambda * ordinal_expected_rank_loss(logits, y_idx, class_ranks))

                (loss / accum_steps).backward()

        running += loss.item() * x.size(0)
        n += x.size(0)

        do_step = (micro_step % accum_steps == 0) or (micro_step == len(loader))
        if do_step:
            update_step += 1

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
            if task == "ordinal_mse":
                # Compute class accuracy by nearest-rank assignment (only meaningful when not mixed)
                if did_mix:
                    batch_acc = 0.0
                else:
                    d = torch.abs(pred_rank.detach().unsqueeze(1) - class_ranks.unsqueeze(0))  # [B,C]
                    pred_idx = d.argmin(dim=1)
                    batch_acc = (pred_idx == y_idx).float().mean().item()
            else:
                pred = logits.argmax(dim=1)
                batch_acc = (pred == y_idx).float().mean().item()

        if step % log_every == 0 or step == 1:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = running / max(n, 1)
            elapsed = time.time() - t0
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                batch_acc=f"{batch_acc*100:.2f}%" if not did_mix else "NA(mix)",
                lr=f"{lr:.2e}",
                accum=f"{accum_steps}",
                upd=f"{update_step}",
                t=f"{elapsed:.0f}s",
            )

    return running / max(n, 1)


# -----------------------------
# Build transforms
# -----------------------------
def build_transforms(img_size: int, mean, std, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.92, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.04, contrast=0.04, saturation=0.02, hue=0.01),
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
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default=None)
    ap.add_argument("--root_dir", default=None)
    ap.add_argument("--img_col", default=None)
    ap.add_argument("--label_col", default=None)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup_epochs", type=int, default=2)

    # Two-stage + progressive resizing
    ap.add_argument("--mix_epochs", type=int, default=8)

    # UPGRADE: smaller default resolution for stability and speed
    ap.add_argument("--img_size", type=int, default=448, help="Stage A img size")
    ap.add_argument("--img_size_ft", type=int, default=672, help="Stage B img size (fine-tune)")

    # UPDATED: standard Stage A LR
    ap.add_argument("--lr", type=float, default=3e-4)
    # UPDATED (Version 8): safer fine-tune LR
    ap.add_argument("--lr_ft", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--out", default="clarity_best.pt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_decode_retries", type=int, default=10)

    # label normalization defaults ON
    ap.add_argument("--normalize_labels", action="store_true", default=True)
    ap.add_argument("--no_normalize_labels", action="store_true")
    ap.add_argument("--merge_si3_to_si2", action="store_true", default=True)

    # crop cache defaults ON
    ap.add_argument("--crop_cache", action="store_true", default=True)
    ap.add_argument("--no_crop_cache", action="store_true")
    # UPDATED (Version 8): avoid conflicts with older cache
    ap.add_argument("--cache_dir", default="_clarity_cache_v8")

    # sampler Stage A (and now also Stage B)
    ap.add_argument("--sampler", action="store_true", default=True)
    ap.add_argument("--no_sampler", action="store_true")

    # MixUp/CutMix Stage A (kept; for ordinal regression we use scalar MixUp)
    ap.add_argument("--mix_p", type=float, default=0.6)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--cutmix_alpha", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=50)

    # fine-tune label smoothing
    ap.add_argument("--label_smoothing_ft", type=float, default=0.05)

    # EMA + clip
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # clarity-specific ordinal regularization (kept for compatibility)
    ap.add_argument("--ordinal_lambda", type=float, default=0.12)

    # reports
    ap.add_argument("--per_class", action="store_true", default=True)
    ap.add_argument("--no_per_class", action="store_true")
    ap.add_argument("--print_cm", action="store_true", default=True)
    ap.add_argument("--no_print_cm", action="store_true")
    ap.add_argument("--cm_normalize", action="store_true", default=True)
    ap.add_argument("--no_cm_normalize", action="store_true")
    ap.add_argument("--top_confusions", type=int, default=2)

    # TTA
    ap.add_argument("--tta_mode", default="fivecrop", choices=["none", "flip", "fivecrop", "fivecrop_flip"])

    # NEW: task + gradient accumulation
    ap.add_argument("--task", default="ordinal_mse", choices=["ordinal_mse", "classification"])
    ap.add_argument("--effective_batch", type=int, default=64, help="Target effective batch via accumulation")
    ap.add_argument("--accum_steps", type=int, default=None, help="If None, auto = ceil(effective_batch / batch)")

    # NEW: (optional) loss weighting for imbalance
    # UPDATED: default False (scale handles logic; don't penalize by frequency)
    ap.add_argument("--use_loss_weights", action="store_true", default=False)
    ap.add_argument("--no_loss_weights", action="store_true")

    # Version 8: focal gamma for regression
    ap.add_argument("--focal_gamma", type=float, default=0.5, help="FocalMSE gamma for ordinal regression")

    args = ap.parse_args()

    if args.no_normalize_labels:
        args.normalize_labels = False
    if args.no_crop_cache:
        args.crop_cache = False
    if args.no_per_class:
        args.per_class = False
    if args.no_print_cm:
        args.print_cm = False
    if args.no_cm_normalize:
        args.cm_normalize = False
    if args.no_sampler:
        args.sampler = False
    if args.no_ema:
        args.ema = False
    if args.no_loss_weights:
        args.use_loss_weights = False

    if args.accum_steps is None:
        args.accum_steps = max(1, int(math.ceil(args.effective_batch / max(1, args.batch))))

    seed_all(args.seed)

    # auto-download if needed
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

    img_col = args.img_col or guess_column(cols, ["image", "img", "path", "file", "filename"])
    label_col = args.label_col or guess_column(cols, ["clarity"])
    if img_col is None or label_col is None:
        raise ValueError(f"לא הצלחתי לזהות עמודות. columns={cols}. תן --img_col --label_col")

    df = df.dropna(subset=[img_col, label_col]).copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    if args.normalize_labels:
        df[label_col] = df[label_col].apply(lambda s: normalize_clarity_label(s, merge_si3_to_si2=args.merge_si3_to_si2))
        print("[INFO] Label normalization: ON")
    else:
        print("[INFO] Label normalization: OFF")

    # crop cache
    if args.crop_cache:
        cache_root = os.path.join(os.path.dirname(args.csv), args.cache_dir)
        print(f"[INFO] Crop cache: ON -> {cache_root}")
        df = preprocess_crops(df, img_col=img_col, root_dir=args.root_dir, cache_dir=cache_root)
        args.root_dir = cache_root
    else:
        print("[INFO] Crop cache: OFF")

    # classes (ordinal order first)
    order = ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]
    present = set(df[label_col].unique().tolist())
    classes = [c for c in order if c in present]
    extras = sorted(list(present - set(classes)))
    classes = classes + extras

    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    num_classes = len(classes)

    print(f"[INFO] img_col={img_col} label_col={label_col} num_classes={num_classes}")
    print(f"[INFO] classes={classes}")
    print("[INFO] BACKBONE=ConvNeXt-Tiny | weights=ConvNeXt_Tiny_Weights.DEFAULT (ImageNet)")
    print(f"[INFO] Task={args.task} | GradAccum: batch={args.batch} effective_batch={args.effective_batch} accum_steps={args.accum_steps}")
    print(f"[INFO] Two-stage: mix_epochs={min(args.mix_epochs, args.epochs)} | lr={args.lr} | lr_ft={args.lr_ft}")
    print(f"[INFO] Resize: stageA={args.img_size} | stageB(ft)={args.img_size_ft}")
    print(f"[INFO] MixUp/CutMix (Stage A): p={args.mix_p} mixup_alpha={args.mixup_alpha} cutmix_alpha={args.cutmix_alpha}")
    print(f"[INFO] Fine-tune label_smoothing={args.label_smoothing_ft}")
    print(f"[INFO] Ordinal lambda={args.ordinal_lambda} (ignored in ordinal_mse)")
    print(f"[INFO] EMA={'ON' if args.ema else 'OFF'} | grad_clip={args.grad_clip}")
    print(f"[INFO] TTA mode={args.tta_mode}")
    print(f"[INFO] Loss weights default={'ON' if args.use_loss_weights else 'OFF'}")
    if args.task == "ordinal_mse":
        print(f"[INFO] FocalMSELoss: gamma={args.focal_gamma}")

    # split
    train_idx, val_idx = stratified_split_indices(df[label_col].tolist(), args.val_split, args.seed)
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # normalization
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std

    # Stage A transforms
    train_tfm_A = build_transforms(args.img_size, mean, std, train=True)
    # Validate at stageB res (same default)
    val_tfm = build_transforms(args.img_size_ft, mean, std, train=False)

    # ranks aligned with classes (mildly adjusted scale)
    class_ranks_cpu = build_clarity_rank(classes)  # CPU copy for dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_ranks = class_ranks_cpu.to(device)

    ds_train = ClarityDataset(
        df_train, img_col, label_col, args.root_dir, label2idx,
        tfm=train_tfm_A,
        max_decode_retries=args.max_decode_retries,
        class_ranks=class_ranks_cpu,
        return_rank=(args.task == "ordinal_mse"),
    )
    ds_val = ClarityDataset(
        df_val, img_col, label_col, args.root_dir, label2idx,
        tfm=val_tfm,
        max_decode_retries=args.max_decode_retries,
        class_ranks=class_ranks_cpu,
        return_rank=(args.task == "ordinal_mse"),
    )

    # sampler (Stage A AND Stage B now)
    sampler = None
    if args.sampler:
        y_train = df_train[label_col].tolist()
        counts: Dict[str, int] = {}
        for y in y_train:
            counts[y] = counts.get(y, 0) + 1
        sample_weights = [1.0 / counts[y] for y in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print("[INFO] Sampler: WeightedRandomSampler ON (Stage A + Stage B).")
    else:
        print("[INFO] Sampler: OFF.")

    # optional loss weights (usually keep OFF if sampler is ON; available if you disable sampler)
    class_weights = None
    if args.use_loss_weights:
        counts_per_idx = torch.zeros(num_classes, dtype=torch.float32)
        for y_str in df_train[label_col].tolist():
            counts_per_idx[label2idx[y_str]] += 1.0
        counts_per_idx = counts_per_idx.clamp_min(1.0)
        w = 1.0 / counts_per_idx
        w = w / w.mean()
        class_weights = w.to(device)
        print("[INFO] Loss weights: ON (inverse freq, mean-normalized).")
    else:
        print("[INFO] Loss weights: OFF.")

    def make_train_loader(use_sampler: bool) -> DataLoader:
        s = sampler if (use_sampler and sampler is not None) else None
        return DataLoader(
            ds_train,
            batch_size=args.batch,
            shuffle=(s is None),
            sampler=s,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )

    dl_train = make_train_loader(use_sampler=args.sampler)

    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = ClarityNet(num_classes=num_classes, dropout=args.dropout, task=args.task).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    autocast_fn, scaler, amp_available = get_amp(device)
    use_amp = amp_available and (not args.no_amp)

    # UPDATED (Version 8): use FocalMSELoss for ordinal regression
    criterion_hard = FocalMSELoss(gamma=args.focal_gamma, reduction="mean") if args.task == "ordinal_mse" else nn.CrossEntropyLoss()

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    best_f1 = -1.0
    best_acc = -1.0

    mix_epochs = min(args.mix_epochs, args.epochs)
    fine_tune_started = False

    # Scheduler for Stage A (steps per optimizer update, not per micro-batch)
    micro_steps_per_epoch = len(dl_train)
    update_steps_per_epoch = int(math.ceil(micro_steps_per_epoch / max(1, args.accum_steps)))
    total_steps_A = update_steps_per_epoch * mix_epochs
    warmup_steps_A = update_steps_per_epoch * max(0, args.warmup_epochs)
    sched = WarmupCosine(optimizer, warmup_steps=warmup_steps_A, total_steps=max(1, total_steps_A))
    sched.set_lr(0)

    for epoch in range(1, args.epochs + 1):
        # Switch to fine-tune
        if (not fine_tune_started) and (epoch == mix_epochs + 1):
            fine_tune_started = True

            # KEEP sampler in Stage B to prevent minority collapse
            dl_train = make_train_loader(use_sampler=args.sampler)
            ds_train.tfm = build_transforms(args.img_size_ft, mean, std, train=True)

            for pg in optimizer.param_groups:
                pg["lr"] = args.lr_ft

            micro_steps_per_epoch = len(dl_train)
            update_steps_per_epoch = int(math.ceil(micro_steps_per_epoch / max(1, args.accum_steps)))
            remaining_epochs = max(1, args.epochs - mix_epochs)
            total_steps_B = update_steps_per_epoch * remaining_epochs
            warmup_steps_B = 0
            sched = WarmupCosine(optimizer, warmup_steps=warmup_steps_B, total_steps=max(1, total_steps_B))
            sched.set_lr(0)

            print(
                f"[STAGE] Fine-tune starts at epoch {epoch}: sampler={'ON' if args.sampler else 'OFF'}, "
                f"train img_size -> {args.img_size_ft}, lr -> {args.lr_ft}, accum_steps -> {args.accum_steps}"
            )

        if epoch <= mix_epochs:
            cur_mix_p = args.mix_p
            cur_mixup = args.mixup_alpha
            cur_cutmix = args.cutmix_alpha
            cur_ls = 0.0
        else:
            cur_mix_p = 0.0
            cur_mixup = 0.0
            cur_cutmix = 0.0
            cur_ls = max(0.0, float(args.label_smoothing_ft))

        train_loss = train_one_epoch(
            model=model,
            loader=dl_train,
            device=device,
            optimizer=optimizer,
            autocast_fn=autocast_fn,
            scaler=scaler,
            use_amp=use_amp,
            num_classes=num_classes,
            mix_p=cur_mix_p,
            mixup_alpha=cur_mixup,
            cutmix_alpha=cur_cutmix,
            label_smoothing=cur_ls,
            log_every=args.log_every,
            scheduler=sched,
            ema=ema,
            grad_clip=args.grad_clip,
            ordinal_lambda=args.ordinal_lambda,
            class_ranks=class_ranks,

            task=args.task,
            accum_steps=args.accum_steps,
            class_weights=class_weights,
            focal_gamma=args.focal_gamma,
        )

        # Validate using EMA weights if enabled
        if ema is not None:
            ema.apply_shadow(model)

        lr_now = optimizer.param_groups[0]["lr"]
        val_metrics, conf = evaluate(
            model, dl_val, device, criterion_hard,
            tta_mode="none",
            lr=lr_now,
            show_pbar=True,
            desc="val",
            task=args.task,
            class_ranks=class_ranks,
            classes=classes,
        )

        if ema is not None:
            ema.restore(model)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_acc={val_metrics.acc*100:.2f}% | "
            f"val_macroF1={val_metrics.macro_f1:.4f}"
        )

        if args.per_class:
            print_per_class_report(conf, classes, title=f"[VAL] Per-class @ epoch {epoch:02d}")

        if args.print_cm:
            print_confusion_matrix(conf, classes, normalize=False, title=f"[VAL] Confusion matrix (counts) @ epoch {epoch:02d}")
            if args.cm_normalize:
                print_confusion_matrix(conf, classes, normalize=True, title=f"[VAL] Confusion matrix (row-normalized) @ epoch {epoch:02d}")
            if args.top_confusions > 0:
                print_top_confusions(conf, classes, k=args.top_confusions, title=f"[VAL] Top confusions @ epoch {epoch:02d}")

        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
            best_acc = val_metrics.acc
            ckpt = {
                "model": model.state_dict(),
                "ema": (ema.shadow if ema is not None else None),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "img_col": img_col,
                "label_col": label_col,
                "root_dir": args.root_dir,
                "classes": classes,
                "backbone": "convnext_tiny",
                "weights": "ConvNeXt_Tiny_Weights.DEFAULT",
                "normalize_labels": bool(args.normalize_labels),
                "crop_cache": bool(args.crop_cache),
                "task": args.task,
                "grad_accum": {
                    "batch": args.batch,
                    "effective_batch": args.effective_batch,
                    "accum_steps": args.accum_steps,
                },
                "two_stage": {
                    "mix_epochs": mix_epochs,
                    "lr": args.lr,
                    "lr_ft": args.lr_ft,
                    "img_size": args.img_size,
                    "img_size_ft": args.img_size_ft,
                },
                "mixup_cutmix_stageA": {"p": args.mix_p, "mixup_alpha": args.mixup_alpha, "cutmix_alpha": args.cutmix_alpha},
                "ema_decay": args.ema_decay if args.ema else None,
                "ordinal_lambda": args.ordinal_lambda,
                "sampler": bool(args.sampler),
                "use_loss_weights": bool(args.use_loss_weights),
                "focal_mse": {"gamma": args.focal_gamma} if args.task == "ordinal_mse" else None,
                "cache_dir": args.cache_dir,
            }
            torch.save(ckpt, args.out)
            print(f"[SAVE] best -> {args.out} (macroF1={best_f1:.4f}, acc={best_acc*100:.2f}%)")

    # Final TTA evaluation (EMA weights if exists)
    if ema is not None:
        ema.apply_shadow(model)

    lr_now = optimizer.param_groups[0]["lr"]
    tta_metrics, tta_conf = evaluate(
        model, dl_val, device, criterion_hard,
        tta_mode=args.tta_mode,
        lr=lr_now,
        show_pbar=True,
        desc=f"val_tta({args.tta_mode})",
        task=args.task,
        class_ranks=class_ranks,
        classes=classes,
    )

    if ema is not None:
        ema.restore(model)

    print(
        f"[TTA:{args.tta_mode}] val_loss={tta_metrics.loss:.4f} | "
        f"val_acc={tta_metrics.acc*100:.2f}% | "
        f"val_macroF1={tta_metrics.macro_f1:.4f}"
    )
    if args.per_class:
        print_per_class_report(tta_conf, classes, title=f"[TTA:{args.tta_mode}] Per-class on VAL (final)")
    if args.print_cm:
        print_confusion_matrix(tta_conf, classes, normalize=False, title=f"[TTA:{args.tta_mode}] Confusion matrix (counts) on VAL (final)")
        if args.cm_normalize:
            print_confusion_matrix(tta_conf, classes, normalize=True, title=f"[TTA:{args.tta_mode}] Confusion matrix (row-normalized) on VAL (final)")
        if args.top_confusions > 0:
            print_top_confusions(tta_conf, classes, k=args.top_confusions, title=f"[TTA:{args.tta_mode}] Top confusions on VAL (final)")

    print(f"[DONE] best_macroF1={best_f1:.4f} | best_acc={best_acc*100:.2f}% | saved={args.out}")


if __name__ == "__main__":
    main()
