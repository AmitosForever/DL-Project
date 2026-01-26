import os
import re
import io
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
# AMP helpers
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
def normalize_fluorescence_label(x: Any) -> Optional[str]:
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

    # remove common color words
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
    if label_mode == "5class":
        labels = ["None", "Faint", "Medium", "Strong", "Very Strong"]
        collapse = {k: k for k in labels}
        return labels, collapse

    if label_mode == "4class":
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

    def make_full(v: Any) -> Optional[str]:
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


def detect_tabular_columns(df: pd.DataFrame, user_cols: str) -> List[str]:
    if user_cols.strip():
        cols = [c.strip() for c in user_cols.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Tabular columns not found in CSV: {missing}")
        return cols

    # common names in diamond datasets
    wanted = ["carat", "color", "clarity", "cut"]
    found = []
    for w in wanted:
        hits = [c for c in df.columns if c.lower() == w]
        if hits:
            found.append(hits[0])
        else:
            # fuzzy: contains keyword
            hits2 = [c for c in df.columns if w in c.lower()]
            if hits2:
                found.append(hits2[0])
    # dedupe
    out = []
    seen = set()
    for c in found:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


# -----------------------------
# Robust image loading + optional background handling
# -----------------------------
def try_load_rgb(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def simple_background_remove_and_crop(img: Image.Image, bg_rgb=(128, 128, 128), margin: float = 0.10) -> Image.Image:
    """
    Cheap heuristic:
      - mask bright-ish pixels (diamond usually brighter than background)
      - crop to bounding box
      - replace background with neutral gray
    This is NOT perfect segmentation. It's a "stop the model from reading the studio backdrop" hack.
    """
    arr = np.array(img).astype(np.uint8)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)

    # threshold at a high percentile, but keep it stable
    thr = float(np.percentile(gray, 70.0))
    mask = gray >= thr

    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return img

    x0, y0, x1, y1 = bbox
    h, w = arr.shape[0], arr.shape[1]
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    mx = int(bw * margin)
    my = int(bh * margin)
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w - 1, x1 + mx)
    y1 = min(h - 1, y1 + my)

    crop = arr[y0:y1 + 1, x0:x1 + 1].copy()
    crop_gray = gray[y0:y1 + 1, x0:x1 + 1]
    crop_mask = crop_gray >= thr

    bg = np.zeros_like(crop)
    bg[..., 0] = bg_rgb[0]
    bg[..., 1] = bg_rgb[1]
    bg[..., 2] = bg_rgb[2]

    out = np.where(crop_mask[..., None], crop, bg).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def rembg_remove_and_crop(img: Image.Image, bg_rgb=(128, 128, 128), margin: float = 0.10) -> Image.Image:
    """
    Uses rembg if installed. Falls back to original if not.
    """
    try:
        from rembg import remove  # type: ignore
    except Exception:
        return img

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    inp = buf.getvalue()

    try:
        out_bytes = remove(inp)
        out_img = Image.open(io.BytesIO(out_bytes))
    except Exception:
        return img

    if out_img.mode != "RGBA":
        out_img = out_img.convert("RGBA")

    arr = np.array(out_img).astype(np.uint8)
    alpha = arr[..., 3]
    mask = alpha > 0

    bbox = _bbox_from_mask(mask)
    if bbox is None:
        # composite without cropping
        rgb = arr[..., :3]
        bg = np.zeros_like(rgb)
        bg[..., 0] = bg_rgb[0]
        bg[..., 1] = bg_rgb[1]
        bg[..., 2] = bg_rgb[2]
        comp = np.where(mask[..., None], rgb, bg).astype(np.uint8)
        return Image.fromarray(comp, mode="RGB")

    x0, y0, x1, y1 = bbox
    h, w = arr.shape[0], arr.shape[1]
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    mx = int(bw * margin)
    my = int(bh * margin)
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w - 1, x1 + mx)
    y1 = min(h - 1, y1 + my)

    crop = arr[y0:y1 + 1, x0:x1 + 1].copy()
    crop_mask = crop[..., 3] > 0
    rgb = crop[..., :3]

    bg = np.zeros_like(rgb)
    bg[..., 0] = bg_rgb[0]
    bg[..., 1] = bg_rgb[1]
    bg[..., 2] = bg_rgb[2]

    comp = np.where(crop_mask[..., None], rgb, bg).astype(np.uint8)
    return Image.fromarray(comp, mode="RGB")


# -----------------------------
# Tabular preprocessing
# -----------------------------
@dataclass
class TabularPreprocessor:
    transformer: ColumnTransformer
    feature_dim: int
    numeric_cols: List[str]
    categorical_cols: List[str]

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        X = self.transformer.transform(df)
        # ColumnTransformer may output sparse
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        return X


def build_tabular_preprocessor(df_train: pd.DataFrame, tab_cols: List[str]) -> TabularPreprocessor:
    if not tab_cols:
        raise RuntimeError("No tabular columns specified/detected.")

    # split into numeric/categorical
    numeric_cols = []
    categorical_cols = []
    for c in tab_cols:
        if pd.api.types.is_numeric_dtype(df_train[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    numeric_tf = StandardScaler(with_mean=True, with_std=True)
    cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    transformer.fit(df_train[tab_cols])

    # infer feature_dim
    Xtmp = transformer.transform(df_train[tab_cols].head(10))
    if hasattr(Xtmp, "toarray"):
        Xtmp = Xtmp.toarray()
    feature_dim = int(np.asarray(Xtmp).shape[1])

    return TabularPreprocessor(
        transformer=transformer,
        feature_dim=feature_dim,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


# -----------------------------
# Dataset
# -----------------------------
class DiamondMultiModalDataset(Dataset):
    """
    Returns (image_tensor, tabular_tensor, label_idx)
    Also supports get_sample(i) for Grad-CAM visualization.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        img_col: str,
        label_idx_col: str,
        tab_features: np.ndarray,
        transform,
        bg_mode: str = "none",
        bg_rgb: Tuple[int, int, int] = (128, 128, 128),
        max_retry: int = 30,
    ):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_idx_col = label_idx_col
        self.transform = transform
        self.bg_mode = bg_mode
        self.bg_rgb = bg_rgb
        self.max_retry = int(max_retry)

        if len(tab_features) != len(self.df):
            raise ValueError("tab_features length must match df length.")
        self.tab = torch.tensor(tab_features, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def _preprocess_pil(self, pil: Image.Image) -> Image.Image:
        if self.bg_mode == "none":
            return pil
        if self.bg_mode == "simple":
            return simple_background_remove_and_crop(pil, bg_rgb=self.bg_rgb)
        if self.bg_mode == "rembg":
            return rembg_remove_and_crop(pil, bg_rgb=self.bg_rgb)
        return pil

    def _load_pil(self, path: str) -> Optional[Image.Image]:
        img = try_load_rgb(path)
        if img is None:
            return None
        img = self._preprocess_pil(img)
        return img

    def __getitem__(self, idx: int):
        n = len(self.df)
        for attempt in range(self.max_retry):
            j = idx if attempt == 0 else random.randrange(0, n)
            row = self.df.iloc[j]
            path = str(row[self.img_col])
            pil = self._load_pil(path)
            if pil is None:
                continue
            x = self.transform(pil) if self.transform is not None else pil
            y = int(row[self.label_idx_col])
            t = self.tab[j]
            return x, t, y

        last_path = str(self.df.iloc[idx][self.img_col])
        raise RuntimeError(f"Too many unreadable images encountered. Last tried: {last_path}")

    def get_sample(self, idx: int) -> Tuple[Image.Image, torch.Tensor, torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        path = str(row[self.img_col])
        pil = try_load_rgb(path)
        if pil is None:
            raise RuntimeError(f"Unreadable image for Grad-CAM: {path}")
        pil_proc = self._preprocess_pil(pil.copy())
        x = self.transform(pil_proc) if self.transform is not None else pil_proc
        y = int(row[self.label_idx_col])
        t = self.tab[idx]
        return pil_proc, x, t, y, path


# -----------------------------
# Model / Loss
# -----------------------------
def build_convnext_tiny_backbone(pretrained: bool = True) -> nn.Module:
    if pretrained and _HAS_CONVNEXT_WEIGHTS:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
    else:
        model = convnext_tiny(weights=None)
    return model


class DualStreamNetwork(nn.Module):
    """
    Stream A: ConvNeXt-Tiny image features
    Stream B: tabular MLP
    Fusion: concat then classification head
    """
    def __init__(
        self,
        backbone: nn.Module,
        tab_in_dim: int,
        tab_hidden: int,
        tab_out: int,
        fused_hidden: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = backbone

        # ConvNeXt classifier structure: [LayerNorm2d, Flatten, Linear]
        # We'll reuse LayerNorm2d + Flatten logic to obtain image feature vector.
        self.img_norm = backbone.classifier[0]
        self.img_feat_dim = backbone.classifier[-1].in_features

        # Replace backbone final classifier with identity; we handle head ourselves.
        backbone.classifier[-1] = nn.Identity()

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_in_dim, tab_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden, tab_out),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(self.img_feat_dim + tab_out, fused_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_hidden, num_classes),
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(images)
        x = self.backbone.avgpool(x)          # (B,C,1,1)
        x = self.img_norm(x)                  # LayerNorm2d
        x = torch.flatten(x, 1)               # (B,C)

        t = self.tab_mlp(tabular)
        fused = torch.cat([x, t], dim=1)
        logits = self.head(fused)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


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
class EpochStats:
    loss: float
    acc: float
    macro_f1: float


# -----------------------------
# TTA helpers
# -----------------------------
def tta_logits(model: nn.Module, xb: torch.Tensor, tb: torch.Tensor) -> torch.Tensor:
    """
    4-way TTA:
      - original
      - hflip
      - vflip
      - rot90
    Average logits (soft voting).
    """
    logits_list = []

    logits_list.append(model(xb, tb))

    xb_h = torch.flip(xb, dims=[3])
    logits_list.append(model(xb_h, tb))

    xb_v = torch.flip(xb, dims=[2])
    logits_list.append(model(xb_v, tb))

    xb_r = torch.rot90(xb, k=1, dims=[2, 3])
    logits_list.append(model(xb_r, tb))

    return torch.stack(logits_list, dim=0).mean(dim=0)


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    scaler,
    criterion: nn.Module,
    autocast_ctx_fn,
    max_grad_norm: float,
    num_classes: int,
    desc: str,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    all_true = []
    all_pred = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for xb, tb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        tb = tb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx_fn():
            logits = model(xb, tb)
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
    acc, macro_f1 = compute_metrics(y_true, y_pred, num_classes=num_classes)
    avg_loss = total_loss / max(1, total_seen)
    return EpochStats(loss=avg_loss, acc=acc, macro_f1=macro_f1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    autocast_ctx_fn,
    num_classes: int,
    desc: str,
    use_tta: bool,
) -> Tuple[EpochStats, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_correct = 0
    all_true = []
    all_pred = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for xb, tb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        tb = tb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with autocast_ctx_fn():
            if use_tta:
                logits = tta_logits(model, xb, tb)
            else:
                logits = model(xb, tb)
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
            acc=f"{total_correct / max(1, total_seen):.4f}",
        )

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    acc, macro_f1 = compute_metrics(y_true, y_pred, num_classes=num_classes)
    avg_loss = total_loss / max(1, total_seen)
    return EpochStats(loss=avg_loss, acc=acc, macro_f1=macro_f1), y_true, y_pred


# -----------------------------
# Grad-CAM
# -----------------------------
class GradCAM:
    """
    Minimal Grad-CAM for ConvNeXt feature maps.
    """
    def __init__(self, model: DualStreamNetwork, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._h1 = None
        self._h2 = None

        def fwd_hook(_, __, output):
            self.activations = output

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._h1 = target_layer.register_forward_hook(fwd_hook)
        self._h2 = target_layer.register_full_backward_hook(bwd_hook)

    def close(self):
        if self._h1 is not None:
            self._h1.remove()
        if self._h2 is not None:
            self._h2.remove()

    def __call__(self, images: torch.Tensor, tabular: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images, tabular)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1)[0].item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        A = self.activations  # (B,C,H,W)
        G = self.gradients    # (B,C,H,W)

        weights = G.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = F.relu(cam)

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam  # (B,1,H,W)


def tensor_to_pil_unnormalized(x: torch.Tensor) -> Image.Image:
    """
    x: (3,H,W) normalized with ImageNet mean/std.
    Returns PIL RGB in [0,255].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    y = x * std + mean
    y = torch.clamp(y, 0.0, 1.0)
    y = (y * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(y, mode="RGB")


def overlay_heatmap(base: Image.Image, cam: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    """
    cam: (1,H,W) in [0,1]
    Minimal colormap-ish overlay without extra deps: red channel boost.
    """
    cam_np = cam.squeeze(0).detach().cpu().numpy()
    cam_img = (cam_np * 255.0).astype(np.uint8)

    base_np = np.array(base).astype(np.float32)
    heat = np.zeros_like(base_np)
    heat[..., 0] = cam_img  # red channel

    out = (1.0 - alpha) * base_np + alpha * heat
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


@torch.no_grad()
def visualize_predictions_gradcam(
    model: DualStreamNetwork,
    dataset: DiamondMultiModalDataset,
    device: torch.device,
    out_dir: str,
    labels: List[str],
    num_images: int = 5,
    target_names: Optional[List[str]] = None,
):
    safe_mkdir(out_dir)
    model.eval()

    # pick target layer: last features block
    target_layer = model.backbone.features[-1]
    cam_engine = GradCAM(model, target_layer)
    try:
        idxs = random.sample(range(len(dataset)), k=min(num_images, len(dataset)))
        for k, idx in enumerate(idxs):
            pil_proc, x, t, y, path = dataset.get_sample(idx)

            x = x.unsqueeze(0).to(device)
            t = t.unsqueeze(0).to(device)

            # predicted class
            logits = model(x, t)
            pred = int(torch.argmax(logits, dim=1)[0].item())

            # default targets: predicted; optionally specific class names like ["Medium","Strong"]
            targets = []
            if target_names:
                name_to_idx = {n: i for i, n in enumerate(labels)}
                for nm in target_names:
                    if nm in name_to_idx:
                        targets.append((nm, name_to_idx[nm]))
            if not targets:
                targets.append((f"pred_{labels[pred]}", pred))

            for nm, ci in targets:
                # gradcam needs grads; temporarily enable grad
                torch.set_grad_enabled(True)
                cam = cam_engine(x, t, class_idx=ci)  # (1,1,h,w)
                torch.set_grad_enabled(False)

                # upsample cam to image size
                cam_up = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)[0]
                base = tensor_to_pil_unnormalized(x[0])
                out = overlay_heatmap(base, cam_up, alpha=0.45)

                fn = os.path.join(out_dir, f"gradcam_{k:02d}_{nm}_true_{labels[y]}_pred_{labels[pred]}.jpg")
                out.save(fn, quality=92)
    finally:
        cam_engine.close()
        torch.set_grad_enabled(True)


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

    def resolve_path(v: Any) -> str:
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
    group_col = choose_group_column(df, exclude=[fluor_col, img_col, "_label_idx", "_label_name"])
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
# Color-safe transforms (Protect Color Integrity)
# -----------------------------
def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Fluorescence is color-driven (often blue). We keep brightness/contrast jitter,
    but do NOT mess with hue, and keep saturation jitter near-zero.
    Add RandomAffine to simulate camera geometry without destroying chroma.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.01,  # negligible
            hue=0.0,          # DO NOT shift hue
        ),
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
    parser = argparse.ArgumentParser("Diamond Fluorescence (Multi-Modal ConvNeXt + Tabular) - v4")
    parser.add_argument("--data_root", type=str, default="", help="Dataset root dir. If empty, auto-detect from kagglehub cache.")
    parser.add_argument("--auto_kagglehub", action="store_true", help="Force kagglehub download (if installed).")
    parser.add_argument("--kaggle_dataset", type=str, default="aayushpurswani/diamond-images-dataset", help="kagglehub dataset slug.")
    parser.add_argument("--out_dir", type=str, default="_fluor_out_v4", help="Output directory.")
    parser.add_argument("--fluor_col", type=str, default="", help="Explicit fluorescence column name (optional).")
    parser.add_argument("--img_col", type=str, default="", help="Explicit image path column name (optional).")
    parser.add_argument("--tab_cols", type=str, default="", help="Comma-separated tabular cols. Default tries carat,color,clarity,cut.")

    parser.add_argument("--label_mode", type=str, default="4class", choices=["5class", "4class", "3class", "binary"],
                        help="Label space. Default 4class merges Very Strong into Strong.")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--no_sampler", action="store_true", help="Disable WeightedRandomSampler (default ON).")
    parser.add_argument("--no_focal", action="store_true", help="Disable FocalLoss (default ON).")
    parser.add_argument("--focal_gamma", type=float, default=1.0)

    parser.add_argument(
        "--class_weights",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="auto=ON when sampler OFF, OFF when sampler ON.",
    )

    parser.add_argument("--warmup_epochs", type=int, default=1, help="Epochs to train head-only (freeze backbone).")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience on val macro-F1.")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum improvement for early stopping.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--min_groups_for_group_split", type=int, default=50)

    parser.add_argument("--verify_images", action="store_true",
                        help="Pre-scan and drop unreadable images (slower startup, fewer crashes).")
    parser.add_argument("--max_retry", type=int, default=30,
                        help="How many fallback attempts per sample if an image is unreadable.")

    # Background control
    parser.add_argument("--bg_mode", type=str, default="none", choices=["none", "simple", "rembg"],
                        help="Background removal: none/simple/rembg. rembg requires: pip install rembg")
    parser.add_argument("--bg_rgb", type=str, default="128,128,128",
                        help="Background RGB when bg_mode!=none. Format: R,G,B")

    # Explainability / TTA
    parser.add_argument("--tta", action="store_true", help="Enable TTA in VAL/TEST evaluation.")
    parser.add_argument("--gradcam", action="store_true", help="Save Grad-CAM overlays after training.")
    parser.add_argument("--gradcam_images", type=int, default=5, help="How many Grad-CAM images to save.")
    parser.add_argument("--gradcam_targets", type=str, default="Medium,Strong",
                        help="Comma-separated target class names for Grad-CAM; if missing, uses predicted class.")

    args = parser.parse_args()

    set_seed(args.seed)
    safe_mkdir(args.out_dir)

    use_sampler = not args.no_sampler
    use_focal = not args.no_focal

    bg_rgb = tuple(int(x) for x in args.bg_rgb.split(","))
    if len(bg_rgb) != 3:
        raise RuntimeError("--bg_rgb must be R,G,B (e.g., 128,128,128)")

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

    tab_cols = detect_tabular_columns(df, args.tab_cols)
    if not tab_cols:
        raise RuntimeError("No tabular columns detected. Provide --tab_cols explicitly.")
    print(f"[INFO] Tabular columns: {tab_cols}")

    labels, collapse = build_label_space(args.label_mode)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    def map_to_mode(v) -> Optional[str]:
        k = normalize_fluorescence_label(v)
        if k is None:
            return None
        return collapse.get(k, None)

    df["_label_name"] = df[fluor_col_used].apply(map_to_mode)
    df = df[df["_label_name"].notna()].copy()
    df["_label_idx"] = df["_label_name"].apply(lambda s: label_to_idx[str(s)]).astype(int)

    print(f"[INFO] label_mode={args.label_mode} | num_classes={len(labels)} | labels={labels}")
    dist = df["_label_name"].value_counts().to_dict()
    print("[INFO] Label distribution:")
    for k in labels:
        print(f"  {k}: {int(dist.get(k, 0))}")

    y_all = df["_label_idx"].values.astype(int)

    groups = build_groups_if_possible(df, fluor_col=fluor_col_used, img_col=img_col_used)
    if groups is None:
        print("[INFO] groups: None (stratified split)")
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

    # Build tabular preprocessor on TRAIN ONLY
    tab_proc = build_tabular_preprocessor(df_train, tab_cols)
    X_train = tab_proc.transform_df(df_train[tab_cols])
    X_val = tab_proc.transform_df(df_val[tab_cols])
    X_test = tab_proc.transform_df(df_test[tab_cols])

    print(f"[INFO] Tabular feature dim after encode/scale: {tab_proc.feature_dim} "
          f"(num={len(tab_proc.numeric_cols)}, cat={len(tab_proc.categorical_cols)})")

    train_tf, val_tf = build_transforms(args.img_size)

    train_ds = DiamondMultiModalDataset(
        df_train, img_col=img_col_used, label_idx_col="_label_idx",
        tab_features=X_train, transform=train_tf,
        bg_mode=args.bg_mode, bg_rgb=bg_rgb, max_retry=args.max_retry
    )
    val_ds = DiamondMultiModalDataset(
        df_val, img_col=img_col_used, label_idx_col="_label_idx",
        tab_features=X_val, transform=val_tf,
        bg_mode=args.bg_mode, bg_rgb=bg_rgb, max_retry=args.max_retry
    )
    test_ds = DiamondMultiModalDataset(
        df_test, img_col=img_col_used, label_idx_col="_label_idx",
        tab_features=X_test, transform=val_tf,
        bg_mode=args.bg_mode, bg_rgb=bg_rgb, max_retry=args.max_retry
    )

    num_classes = len(labels)
    train_labels_idx = df_train["_label_idx"].values.astype(int)

    sampler = None
    if use_sampler:
        sampler = build_sampler(train_labels_idx, num_classes=num_classes)
        shuffle = False
        print("[INFO] WeightedRandomSampler: ON")
    else:
        shuffle = True
        print("[INFO] WeightedRandomSampler: OFF")

    if args.class_weights == "on":
        use_class_weights = True
    elif args.class_weights == "off":
        use_class_weights = False
    else:
        use_class_weights = (not use_sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False
    )

    backbone = build_convnext_tiny_backbone(pretrained=True)
    model = DualStreamNetwork(
        backbone=backbone,
        tab_in_dim=tab_proc.feature_dim,
        tab_hidden=256,
        tab_out=128,
        fused_hidden=256,
        dropout=0.2,
        num_classes=num_classes,
    ).to(device)

    class_weights = None
    if use_class_weights:
        class_weights = get_class_weights(train_labels_idx, num_classes=num_classes).to(device)
        print("[INFO] Class weights: ON")
    else:
        print("[INFO] Class weights: OFF (recommended when sampler is ON)")

    if use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
        print(f"[INFO] Loss=FocalLoss(gamma={args.focal_gamma}) | sampler={use_sampler} | class_weights={use_class_weights}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[INFO] Loss=CrossEntropyLoss | sampler={use_sampler} | class_weights={use_class_weights}")

    # Optim: separate LRs for backbone vs rest
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]

    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": args.lr_backbone},
         {"params": other_params, "lr": args.lr_head}],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    autocast_ctx_fn = make_autocast(device)
    scaler = make_grad_scaler(device)

    def set_backbone_trainable(trainable: bool):
        for p in model.backbone.parameters():
            p.requires_grad = trainable

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    meta_path = os.path.join(args.out_dir, "meta.json")

    meta = {
        "data_root": data_root,
        "csv_path": csv_path,
        "fluor_col": fluor_col_used,
        "img_col_used": img_col_used,
        "tab_cols": tab_cols,
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
        "class_weights_mode": args.class_weights,
        "class_weights_used": use_class_weights,
        "bg_mode": args.bg_mode,
        "bg_rgb": list(bg_rgb),
        "tta": bool(args.tta),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved meta: {meta_path}")

    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Warmup: head-only (freeze backbone)
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            set_backbone_trainable(False)
            optimizer.param_groups[0]["lr"] = 0.0
            optimizer.param_groups[1]["lr"] = args.lr_head
            print(f"[INFO] Warmup epoch {epoch}/{args.warmup_epochs}: backbone frozen (lr_backbone=0)")
        else:
            set_backbone_trainable(True)
            optimizer.param_groups[0]["lr"] = args.lr_backbone
            optimizer.param_groups[1]["lr"] = args.lr_head

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            autocast_ctx_fn=autocast_ctx_fn,
            max_grad_norm=args.max_grad_norm,
            num_classes=num_classes,
            desc=f"TRAIN e{epoch}/{args.epochs}",
        )

        val_stats, _, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            autocast_ctx_fn=autocast_ctx_fn,
            num_classes=num_classes,
            desc=f"VAL   e{epoch}/{args.epochs}" + (" (TTA)" if args.tta else ""),
            use_tta=bool(args.tta),
        )

        if not (args.warmup_epochs > 0 and epoch <= args.warmup_epochs):
            scheduler.step()

        dt = time.time() - t0

        print(
            f"[EPOCH {epoch:03d}/{args.epochs}] "
            f"train: loss={train_stats.loss:.4f} acc={train_stats.acc:.4f} f1={train_stats.macro_f1:.4f} | "
            f"val: loss={val_stats.loss:.4f} acc={val_stats.acc:.4f} f1={val_stats.macro_f1:.4f} | "
            f"time={dt:.1f}s"
        )

        if val_stats.macro_f1 > best_f1 + args.min_delta:
            best_f1 = val_stats.macro_f1
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1": best_f1,
                    "args": vars(args),
                    "meta": meta,
                    "tabular": {
                        "numeric_cols": tab_proc.numeric_cols,
                        "categorical_cols": tab_proc.categorical_cols,
                    },
                },
                best_path,
            )
            print(f"[CKPT] Saved best -> {best_path} (val macro-F1={best_f1:.4f})")
        else:
            no_improve += 1
            print(f"[EARLY] no_improve={no_improve}/{args.patience} (best_f1={best_f1:.4f})")
            if no_improve >= args.patience:
                print("[EARLY] Stopping early due to no val macro-F1 improvement.")
                break

    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] Loaded best checkpoint (val macro-F1={ckpt.get('best_macro_f1', float('nan')):.4f})")

    test_stats, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        autocast_ctx_fn=autocast_ctx_fn,
        num_classes=num_classes,
        desc="TEST" + (" (TTA)" if args.tta else ""),
        use_tta=bool(args.tta),
    )

    print(f"[TEST] loss={test_stats.loss:.4f} acc={test_stats.acc:.4f} macro-F1={test_stats.macro_f1:.4f}")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("[TEST] Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("[TEST] Classification report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    report_path = os.path.join(args.out_dir, "test_report.json")
    report = {
        "test_loss": test_stats.loss,
        "test_acc": test_stats.acc,
        "test_macro_f1": test_stats.macro_f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "label_mode": args.label_mode,
        "best_val_macro_f1": float(best_f1),
        "tta": bool(args.tta),
        "bg_mode": args.bg_mode,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved test report: {report_path}")

    if args.gradcam:
        gc_dir = os.path.join(args.out_dir, "gradcam")
        targets = [s.strip() for s in args.gradcam_targets.split(",") if s.strip()]
        print(f"[INFO] Saving Grad-CAM overlays to: {gc_dir}")
        visualize_predictions_gradcam(
            model=model,
            dataset=test_ds,
            device=device,
            out_dir=gc_dir,
            labels=labels,
            num_images=args.gradcam_images,
            target_names=targets if targets else None,
        )
        print("[INFO] Grad-CAM done.")


if __name__ == "__main__":
    main()
