import os
import re
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

# allow partially-truncated JPEGs to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

try:
    from torchvision.models import convnext_tiny
    try:
        from torchvision.models import ConvNeXt_Tiny_Weights
        _HAS_CONVNEXT_WEIGHTS = True
    except Exception:
        _HAS_CONVNEXT_WEIGHTS = False
except Exception as e:
    raise RuntimeError("torchvision convnext_tiny is not available. Please upgrade torchvision.") from e

try:
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
except Exception as e:
    raise RuntimeError("scikit-learn is required: pip install scikit-learn") from e

try:
    from tqdm import tqdm
except Exception as e:
    raise RuntimeError("tqdm is required: pip install tqdm") from e


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def path_exists(p: str) -> bool:
    try:
        return os.path.isfile(p)
    except Exception:
        return False


def parse_kaggle_slug(slug: str) -> Tuple[str, str]:
    parts = slug.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid kaggle slug: {slug}")
    return parts[0], parts[1]


def auto_find_kagglehub_cache_root(kaggle_dataset: str) -> Optional[str]:
    owner, dset = parse_kaggle_slug(kaggle_dataset)
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".cache", "kagglehub", "datasets", owner, dset, "versions"),
        os.path.join(home, ".kagglehub", "datasets", owner, dset, "versions"),
    ]
    for versions_dir in candidates:
        if not os.path.isdir(versions_dir):
            continue
        subdirs = []
        try:
            for name in os.listdir(versions_dir):
                full = os.path.join(versions_dir, name)
                if os.path.isdir(full) and name.isdigit():
                    subdirs.append((int(name), full))
        except Exception:
            continue
        if subdirs:
            subdirs.sort(key=lambda x: x[0])
            return subdirs[-1][1]
    return None


def try_download_with_kagglehub(kaggle_dataset: str) -> Optional[str]:
    try:
        import kagglehub  # type: ignore
    except Exception:
        return None
    try:
        root = kagglehub.dataset_download(kaggle_dataset)
        if root and os.path.isdir(root):
            return root
    except Exception:
        return None
    return None


def resolve_data_root(user_data_root: str, kaggle_dataset: str, force_download: bool) -> str:
    if user_data_root:
        if os.path.isdir(user_data_root):
            return user_data_root
        raise RuntimeError(f"data_root does not exist: {user_data_root}")

    for env_name in ["DIAMOND_DATA_ROOT", "KAGGLEHUB_DATASET_ROOT"]:
        v = os.environ.get(env_name, "").strip()
        if v and os.path.isdir(v):
            return v

    if force_download:
        dl = try_download_with_kagglehub(kaggle_dataset)
        if dl is not None:
            return dl

    cached = auto_find_kagglehub_cache_root(kaggle_dataset)
    if cached is not None:
        return cached

    dl = try_download_with_kagglehub(kaggle_dataset)
    if dl is not None:
        return dl

    raise RuntimeError(
        "Could not resolve dataset root automatically.\n"
        "Fix options:\n"
        "  1) Pass --data_root <path_to_versions/1>\n"
        "  2) Install kagglehub: pip install kagglehub\n"
        "  3) Set env DIAMOND_DATA_ROOT to your dataset directory"
    )


# -----------------------------
# AMP helpers (avoid deprecation warnings)
# -----------------------------
def make_autocast(device: torch.device):
    if device.type != "cuda":
        return lambda: nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return lambda: torch.amp.autocast(device_type="cuda")
    return lambda: torch.cuda.amp.autocast()


def make_grad_scaler(device: torch.device):
    enabled = (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# -----------------------------
# Fluorescence normalization
# -----------------------------
CANONICAL_5 = ["None", "Faint", "Medium", "Strong", "Very Strong"]


def normalize_fluorescence_label(x: str) -> Optional[str]:
    """
    Supports:
      - Full: None/Faint/Medium/Strong/Very Strong
      - Abbrev: N, F, M, ST/S, VS/VST
      - Slight: SL, VSL, SLIGHT, VERY SLIGHT -> mapped to Faint
      - 'Medium Blue' etc -> removes color words
    """
    if x is None:
        return None

    s = str(x).strip().lower()
    if s == "" or s == "nan":
        return None

    s = re.sub(r"[\-_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    s = re.sub(r"\b(blue|yellow|white|green|orange|red)\b", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()

    if s in ["n", "no", "none", "nil", "n/a", "na", "0", "non"]:
        return "None"

    if s in ["sl", "vsl", "slight", "very slight", "veryslight", "very-slight"]:
        return "Faint"

    if s in ["f", "fa", "faint", "weak", "low", "1"]:
        return "Faint"
    if s in ["m", "med", "medium", "moderate", "2"]:
        return "Medium"
    if s in ["st", "s", "strong", "high", "3"]:
        return "Strong"
    if s in ["vs", "vst", "v strong", "vstrong", "very strong", "4"]:
        return "Very Strong"

    if "very" in s and "strong" in s:
        return "Very Strong"
    if "strong" in s:
        return "Strong"
    if "medium" in s or s == "med":
        return "Medium"
    if "slight" in s:
        return "Faint"
    if "faint" in s or "weak" in s:
        return "Faint"
    if "none" in s or s.startswith("no "):
        return "None"

    return None


def build_label_space(label_mode: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
      labels: class names in order
      collapse_map: maps CANONICAL_5 -> label in this mode
    """
    if label_mode == "5class":
        labels = ["None", "Faint", "Medium", "Strong", "Very Strong"]
        collapse = {k: k for k in labels}
        return labels, collapse

    if label_mode == "4class":
        # merge Very Strong into Strong
        labels = ["None", "Faint", "Medium", "Strong"]
        collapse = {
            "None": "None",
            "Faint": "Faint",
            "Medium": "Medium",
            "Strong": "Strong",
            "Very Strong": "Strong",
        }
        return labels, collapse

    if label_mode == "3class":
        # None / Faint / (Medium+Strong+VeryStrong)
        labels = ["None", "Faint", "Strong"]
        collapse = {
            "None": "None",
            "Faint": "Faint",
            "Medium": "Strong",
            "Strong": "Strong",
            "Very Strong": "Strong",
        }
        return labels, collapse

    if label_mode == "binary":
        labels = ["None", "Has Fluorescence"]
        collapse = {
            "None": "None",
            "Faint": "Has Fluorescence",
            "Medium": "Has Fluorescence",
            "Strong": "Has Fluorescence",
            "Very Strong": "Has Fluorescence",
        }
        return labels, collapse

    raise ValueError(f"Unknown label_mode: {label_mode}")


# -----------------------------
# Auto-detection columns
# -----------------------------
def detect_fluorescence_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    hits = [c for c in cols if "fluor" in c.lower()]
    if len(hits) == 1:
        return hits[0]

    def score(c: str) -> float:
        ser = df[c].astype(str).fillna("").head(3000)
        normed = [normalize_fluorescence_label(v) for v in ser.tolist()]
        ok = sum(v is not None for v in normed)
        uniq = len(set([v for v in normed if v is not None]))
        return ok + 50.0 * (1.0 if 2 <= uniq <= 6 else 0.0)

    scored = [(c, score(c)) for c in cols]
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored[0][1] <= 0:
        raise RuntimeError("Could not detect fluorescence column. Specify --fluor_col.")
    return scored[0][0]


def detect_image_path_column(df: pd.DataFrame, data_root: str) -> str:
    cols = list(df.columns)
    candidates = []
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in ["image", "img", "path", "file", "filename", "photo", "src"]):
            candidates.append(c)
    if not candidates:
        candidates = cols.copy()

    def make_full(v: str) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return None
        if os.path.isabs(s) and path_exists(s):
            return s
        s2 = s.lstrip("/\\")
        return os.path.join(data_root, s2)

    def score(c: str) -> float:
        ser = df[c].dropna().astype(str)
        if len(ser) == 0:
            return -1.0
        samp = ser.head(500).tolist()
        hits = 0
        for v in samp:
            if str(v).strip().lower().startswith(("http://", "https://")):
                continue
            full = make_full(v)
            if full and path_exists(full) and is_image_file(full):
                hits += 1
        return float(hits)

    scored = [(c, score(c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored[0][1] <= 0:
        scored2 = [(c, score(c)) for c in cols]
        scored2.sort(key=lambda x: x[1], reverse=True)
        if scored2[0][1] <= 0:
            raise RuntimeError("Could not detect image path column. Specify --img_col.")
        return scored2[0][0]
    return scored[0][0]


def choose_group_column(df: pd.DataFrame, exclude: List[str]) -> Optional[str]:
    """
    Only choose truly ID/URL-like columns by NAME.
    Avoid picking target / image path / generic low-cardinality columns.
    """
    exclude_l = set([c.lower() for c in exclude])
    keywords = ["url", "diamond_id", "listing_id", "product_id", "sku", "certificate", "cert", "report", "inventory", "id"]
    cols = list(df.columns)

    candidates = []
    for c in cols:
        lc = c.lower()
        if lc in exclude_l:
            continue
        if any(kw in lc for kw in keywords):
            candidates.append(c)

    if not candidates:
        return None

    best = None
    best_score = -1e9
    n = float(len(df))

    for c in candidates:
        ser = df[c]
        miss = float(ser.isna().mean())
        nunique = float(ser.nunique(dropna=True))
        score = (1.0 - miss) * 100.0 + (nunique / max(1.0, n)) * 50.0
        if score > best_score:
            best_score = score
            best = c

    return best


# -----------------------------
# Robust image loading
# -----------------------------
def try_load_rgb(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, ValueError):
        return None


# -----------------------------
# Dataset
# -----------------------------
class DiamondFluorescenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, label_idx_col: str, transform=None, max_retry: int = 20):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_idx_col = label_idx_col
        self.transform = transform
        self.max_retry = max_retry

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        n = len(self.df)
        for attempt in range(self.max_retry):
            j = idx if attempt == 0 else random.randrange(0, n)
            row = self.df.iloc[j]
            path = row[self.img_col]
            img = try_load_rgb(path)
            if img is None:
                continue
            if self.transform is not None:
                img = self.transform(img)
            y = int(row[self.label_idx_col])
            return img, y

        last_path = str(self.df.iloc[idx][self.img_col])
        raise RuntimeError(f"Too many unreadable images encountered. Last tried: {last_path}")


# -----------------------------
# Model / Loss
# -----------------------------
def build_convnext_tiny_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained and _HAS_CONVNEXT_WEIGHTS:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
    else:
        model = convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0
    )
    return acc, macro_f1


@dataclass
class TrainStats:
    loss: float
    acc: float
    macro_f1: float


@torch.no_grad()
def evaluate_with_progress(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    autocast_ctx_fn,
    num_classes: int,
    desc: str,
) -> Tuple[TrainStats, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    all_true = []
    all_pred = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with autocast_ctx_fn():
            logits = model(xb)
            loss = criterion(logits, yb)

        bs = int(yb.size(0))
        total_loss += float(loss.detach().cpu().item()) * bs
        total_seen += bs

        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred == yb).sum().item())

        all_true.append(yb.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

        pbar.set_postfix(
            loss=f"{total_loss / max(1, total_seen):.4f}",
            acc=f"{total_correct / max(1, total_seen):.4f}"
        )

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    acc, macro_f1 = compute_metrics(y_true, y_pred, num_classes=num_classes)
    avg_loss = total_loss / max(1, total_seen)
    return TrainStats(loss=avg_loss, acc=acc, macro_f1=macro_f1), y_true, y_pred


def train_one_epoch_with_progress(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    scaler,
    criterion: nn.Module,
    autocast_ctx_fn,
    max_grad_norm: float,
    desc: str,
) -> TrainStats:
    model.train()
    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    all_true = []
    all_pred = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx_fn():
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        bs = int(yb.size(0))
        total_loss += float(loss.detach().cpu().item()) * bs
        total_seen += bs

        pred = torch.argmax(logits.detach(), dim=1)
        total_correct += int((pred == yb).sum().item())

        all_true.append(yb.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

        lr0 = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
        pbar.set_postfix(
            loss=f"{total_loss / max(1, total_seen):.4f}",
            acc=f"{total_correct / max(1, total_seen):.4f}",
            lr=f"{lr0:.2e}"
        )

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    num_classes = int(max(y_true.max(), y_pred.max()) + 1) if len(y_true) else 1
    acc, macro_f1 = compute_metrics(y_true, y_pred, num_classes=num_classes)
    avg_loss = total_loss / max(1, total_seen)
    return TrainStats(loss=avg_loss, acc=acc, macro_f1=macro_f1)


def get_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def build_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights_per_class = 1.0 / counts
    weights_per_sample = weights_per_class[labels]
    weights_per_sample = torch.tensor(weights_per_sample, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_per_sample, num_samples=len(weights_per_sample), replacement=True)


# -----------------------------
# Data prep
# -----------------------------
def resolve_csv_path(data_root: str) -> str:
    cand = os.path.join(data_root, "web_scraped", "diamond_data.csv")
    if os.path.isfile(cand):
        return cand
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower() == "diamond_data.csv":
                return os.path.join(root, f)
    raise RuntimeError("Could not find diamond_data.csv under data_root.")


def clean_and_build_dataframe(
    csv_path: str,
    data_root: str,
    fluor_col: Optional[str],
    img_col: Optional[str],
    verify_images: bool,
) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise RuntimeError("CSV loaded but is empty.")

    if fluor_col is None:
        fluor_col = detect_fluorescence_column(df)
    if img_col is None:
        img_col = detect_image_path_column(df, data_root)

    raw_counts = (
        df[fluor_col]
        .astype(str)
        .fillna("")
        .str.strip()
        .replace({"nan": ""})
        .value_counts()
        .head(20)
    )
    print("[DEBUG] Top raw fluorescence values:")
    print(raw_counts.to_string())

    df = df.copy()
    df["_fluor_norm"] = df[fluor_col].apply(normalize_fluorescence_label)

    def resolve_path(v: str) -> str:
        s = str(v).strip()
        if os.path.isabs(s) and path_exists(s):
            return s
        s2 = s.lstrip("/\\")
        return os.path.join(data_root, s2)

    df["_img_full"] = df[img_col].apply(resolve_path)

    mask = (
        df["_fluor_norm"].notna()
        & df["_img_full"].apply(path_exists)
        & df["_img_full"].apply(is_image_file)
    )
    df = df[mask].copy()

    df[fluor_col] = df["_fluor_norm"]
    df["_img_path_for_ds"] = df["_img_full"]
    df = df.drop(columns=["_fluor_norm", "_img_full"], errors="ignore")

    if verify_images:
        ok = []
        pbar = tqdm(df["_img_path_for_ds"].tolist(), desc="VERIFY_IMAGES", leave=False)
        for p in pbar:
            ok.append(try_load_rgb(p) is not None)
        df = df[np.array(ok, dtype=bool)].copy()

    if len(df) < 50:
        raise RuntimeError(f"After cleaning, too few samples remain: {len(df)}.")

    return df, fluor_col, "_img_path_for_ds"


def build_groups_if_possible(df: pd.DataFrame, fluor_col: str, img_col: str) -> Optional[np.ndarray]:
    group_col = choose_group_column(df, exclude=[fluor_col, img_col, "_label_idx"])
    if group_col is None:
        return None

    ser = df[group_col].astype(str).fillna("")
    nunique = ser.nunique()
    if nunique <= 1:
        return None

    n = len(ser)
    if nunique >= int(0.99 * n):
        return None

    return ser.values.astype(str)


def split_indices_auto(
    y: np.ndarray,
    groups: Optional[np.ndarray],
    seed: int,
    test_size: float,
    val_size: float,
    min_groups_for_group_split: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    y = np.asarray(y).astype(int)

    if groups is not None:
        ug = np.unique(groups)
        if len(ug) >= min_groups_for_group_split:
            try:
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                trainval_idx, test_idx = next(gss.split(idx, y, groups=groups))

                rel_val = val_size / max(1e-9, (1.0 - test_size))
                gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
                tr_rel, va_rel = next(gss2.split(trainval_idx, y[trainval_idx], groups=groups[trainval_idx]))
                train_idx = trainval_idx[tr_rel]
                val_idx = trainval_idx[va_rel]

                overall = set(np.unique(y).tolist())
                train_labels = set(np.unique(y[train_idx]).tolist())
                if train_labels != overall:
                    print("[WARN] Group split would exclude some classes from TRAIN. Falling back to STRATIFIED split.")
                else:
                    return train_idx, val_idx, test_idx
            except Exception as e:
                print(f"[WARN] Group split failed ({e}). Falling back to STRATIFIED split.")
        else:
            print(f"[INFO] Not using group split: unique_groups={len(ug)} < min_groups_for_group_split={min_groups_for_group_split}.")

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idx, y))

    rel_val = val_size / max(1e-9, (1.0 - test_size))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr_rel, va_rel = next(sss2.split(trainval_idx, y[trainval_idx]))

    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]
    return train_idx, val_idx, test_idx


# -----------------------------
# Transforms
# -----------------------------
def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02)], p=0.35),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, val_tf


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Diamond Fluorescence Classifier (ConvNeXt-Tiny) - v2 (robust + focal + sampler)")
    parser.add_argument("--data_root", type=str, default="", help="Dataset root dir. If empty, auto-detect from kagglehub cache.")
    parser.add_argument("--auto_kagglehub", action="store_true", help="Force kagglehub download (if installed).")
    parser.add_argument("--kaggle_dataset", type=str, default="aayushpurswani/diamond-images-dataset", help="kagglehub dataset slug.")
    parser.add_argument("--out_dir", type=str, default="_fluor_out_v2", help="Output directory.")
    parser.add_argument("--fluor_col", type=str, default="", help="Explicit fluorescence column name (optional).")
    parser.add_argument("--img_col", type=str, default="", help="Explicit image path column name (optional).")

    parser.add_argument("--label_mode", type=str, default="4class", choices=["5class", "4class", "3class", "binary"],
                        help="Label space. Default 4class merges Very Strong into Strong.")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    # v2 defaults: sampler + focal ON (disable with flags)
    parser.add_argument("--no_sampler", action="store_true", help="Disable WeightedRandomSampler (v2 default is ON).")
    parser.add_argument("--no_focal", action="store_true", help="Disable FocalLoss (v2 default is ON).")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--min_groups_for_group_split", type=int, default=50)

    parser.add_argument("--verify_images", action="store_true",
                        help="Pre-scan all images and drop unreadable ones (slower startup but cleaner).")
    parser.add_argument("--max_retry", type=int, default=20,
                        help="How many fallback attempts per sample if an image is unreadable.")

    args = parser.parse_args()

    set_seed(args.seed)
    safe_mkdir(args.out_dir)

    use_sampler = not args.no_sampler
    use_focal = not args.no_focal

    data_root = resolve_data_root(args.data_root.strip(), args.kaggle_dataset, force_download=args.auto_kagglehub)
    print(f"[INFO] data_root: {data_root}")

    csv_path = resolve_csv_path(data_root)
    print(f"[INFO] CSV: {csv_path}")

    fluor_col = args.fluor_col.strip() or None
    img_col = args.img_col.strip() or None

    df, fluor_col_used, img_col_used = clean_and_build_dataframe(
        csv_path, data_root, fluor_col, img_col, verify_images=args.verify_images
    )

    print(f"[INFO] Detected fluorescence column: {fluor_col_used}")
    print(f"[INFO] Using image path column: {img_col_used}")
    print(f"[INFO] Samples after cleaning: {len(df)}")

    labels, collapse = build_label_space(args.label_mode)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # normalize to CANONICAL_5 then collapse to requested label space
    def map_to_mode(v) -> Optional[str]:
        k = normalize_fluorescence_label(v)
        if k is None:
            return None
        return collapse.get(k, None)

    df["_label_name"] = df[fluor_col_used].apply(map_to_mode)
    df = df[df["_label_name"].notna()].copy()
    df["_label_idx"] = df["_label_name"].apply(lambda s: label_to_idx[str(s)]).astype(int)

    # print distribution in selected mode
    print(f"[INFO] label_mode={args.label_mode} | num_classes={len(labels)} | labels={labels}")
    dist = df["_label_name"].value_counts().to_dict()
    print("[INFO] Label distribution:")
    for k in labels:
        print(f"  {k}: {int(dist.get(k, 0))}")

    y_all = df["_label_idx"].values.astype(int)

    groups = build_groups_if_possible(df, fluor_col=fluor_col_used, img_col=img_col_used)
    if groups is None:
        print("[INFO] groups: None (using stratified split)")
    else:
        print(f"[INFO] groups: using column-like groups | unique={len(np.unique(groups))}")

    train_idx, val_idx, test_idx = split_indices_auto(
        y=y_all,
        groups=groups,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        min_groups_for_group_split=args.min_groups_for_group_split,
    )

    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    df_test = df.iloc[test_idx].copy()

    print(f"[SPLIT] train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")
    for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        d = part["_label_name"].value_counts().to_dict()
        print(f"[SPLIT_DIST] {name}: " + ", ".join([f"{k}={int(d.get(k,0))}" for k in labels]))

    train_tf, val_tf = build_transforms(args.img_size)

    train_ds = DiamondFluorescenceDataset(df_train, img_col=img_col_used, label_idx_col="_label_idx",
                                          transform=train_tf, max_retry=args.max_retry)
    val_ds = DiamondFluorescenceDataset(df_val, img_col=img_col_used, label_idx_col="_label_idx",
                                        transform=val_tf, max_retry=args.max_retry)
    test_ds = DiamondFluorescenceDataset(df_test, img_col=img_col_used, label_idx_col="_label_idx",
                                         transform=val_tf, max_retry=args.max_retry)

    num_classes = len(labels)
    train_labels_idx = df_train["_label_idx"].values.astype(int)

    sampler = None
    if use_sampler:
        sampler = build_sampler(train_labels_idx, num_classes=num_classes)
        shuffle = False
        print("[INFO] Using WeightedRandomSampler (ON).")
    else:
        shuffle = True
        print("[INFO] WeightedRandomSampler: OFF.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_convnext_tiny_model(num_classes=num_classes, pretrained=True).to(device)

    class_weights = get_class_weights(train_labels_idx, num_classes=num_classes).to(device)
    if use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
        print(f"[INFO] Loss=FocalLoss(gamma={args.focal_gamma}) + class weights (ON).")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("[INFO] Loss=CrossEntropyLoss + class weights (Focal OFF).")

    head_params = list(model.classifier.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": args.lr_backbone},
         {"params": head_params, "lr": args.lr_head}],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    autocast_ctx_fn = make_autocast(device)
    scaler = make_grad_scaler(device)

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    meta_path = os.path.join(args.out_dir, "meta.json")

    meta = {
        "data_root": data_root,
        "csv_path": csv_path,
        "fluor_col": fluor_col_used,
        "img_col_used": img_col_used,
        "label_mode": args.label_mode,
        "labels": labels,
        "seed": args.seed,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "verify_images": args.verify_images,
        "max_retry": args.max_retry,
        "use_sampler": use_sampler,
        "use_focal": use_focal,
        "focal_gamma": args.focal_gamma,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved meta: {meta_path}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = train_one_epoch_with_progress(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            autocast_ctx_fn=autocast_ctx_fn,
            max_grad_norm=args.max_grad_norm,
            desc=f"TRAIN e{epoch}/{args.epochs}",
        )

        val_stats, _, _ = evaluate_with_progress(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            autocast_ctx_fn=autocast_ctx_fn,
            num_classes=num_classes,
            desc=f"VAL   e{epoch}/{args.epochs}",
        )

        scheduler.step()
        dt = time.time() - t0

        print(
            f"[EPOCH {epoch:03d}/{args.epochs}] "
            f"train: loss={train_stats.loss:.4f} acc={train_stats.acc:.4f} f1={train_stats.macro_f1:.4f} | "
            f"val: loss={val_stats.loss:.4f} acc={val_stats.acc:.4f} f1={val_stats.macro_f1:.4f} | "
            f"time={dt:.1f}s"
        )

        if val_stats.macro_f1 > best_f1:
            best_f1 = val_stats.macro_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1": best_f1,
                    "args": vars(args),
                    "meta": meta
                },
                best_path,
            )
            print(f"[CKPT] Saved best -> {best_path} (macro-F1={best_f1:.4f})")

    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] Loaded best checkpoint (macro-F1={ckpt.get('best_macro_f1', float('nan')):.4f})")

    test_stats, y_true, y_pred = evaluate_with_progress(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        autocast_ctx_fn=autocast_ctx_fn,
        num_classes=num_classes,
        desc="TEST",
    )

    print(f"[TEST] loss={test_stats.loss:.4f} acc={test_stats.acc:.4f} macro-F1={test_stats.macro_f1:.4f}")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("[TEST] Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    report_path = os.path.join(args.out_dir, "test_report.json")
    report = {
        "test_loss": test_stats.loss,
        "test_acc": test_stats.acc,
        "test_macro_f1": test_stats.macro_f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "label_mode": args.label_mode,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved test report: {report_path}")


if __name__ == "__main__":
    main()
