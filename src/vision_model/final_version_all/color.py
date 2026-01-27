# version7.py
# ------------------------------------------------------------
# DIAMOND tier7 TRAIN (REAL-MAP OR ABORT) + ORDINAL (CORAL) option
# - Saves outputs next to this script in ./_realmap_out
# - Automatically writes eval artifacts to folder(s):
#     _realmap_out/best_eval/ (overwritten on each new best)
#     _realmap_out/eval_history/e{epoch}_bacc{...}_{timestamp}/ (keeps history)
# Artifacts include:
#   report.json, confusion_matrix.csv, per_class_acc.csv
# ------------------------------------------------------------

import os
import re
import time
import json
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
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

try:
    import kagglehub
except Exception as e:
    raise ImportError("Need kagglehub: pip install kagglehub") from e

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

    min_match_ratio: float = 0.80

    unfreeze_at: int = 2
    early_stop_patience: int = 4

    use_ordinal: bool = True

    out_dir_name: str = "_realmap_out"
    model_name: str = "efficientnet_v2_s"  # efficientnet_b0, efficientnet_v2_s


TIER7_ORDER = [
    "Premium_White",
    "Near_Colorless_High",
    "Near_Colorless_Low",
    "Faint_Yellow",
    "Very_Light_Yellow",
    "Light_Yellow",
    "Yellow_LowEnd",
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


def build_image_indexes(image_paths: List[Path]) -> Dict[str, Dict]:
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
    series = df[col].astype(str).fillna("").map(normalize_token)

    by_base = img_index["__by_base__"]
    by_base_lower = img_index["__by_base_lower__"]
    by_num_unique = img_index["__by_num_unique__"]

    n = len(df)
    if n == 0:
        return 0.0, "empty_df"

    sample_n = min(2000, n)

    # A: path or relpath
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

    # B: basename
    matched = 0
    for v in series.head(sample_n):
        b = basename_no_ext(v)
        if b in by_base:
            matched += 1
    ratio_b = matched / sample_n
    if ratio_b > best_ratio:
        best_ratio = ratio_b
        best_name = "basename"

    # C: basename lower
    matched = 0
    for v in series.head(sample_n):
        b = basename_no_ext(v).lower()
        if b in by_base_lower:
            matched += 1
    ratio_c = matched / sample_n
    if ratio_c > best_ratio:
        best_ratio = ratio_c
        best_name = "basename_lower"

    # D: numeric token unique
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
    img = np.clip(img_rgb, 0.0, 1.0)
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p) + eps
    illum = illum / (np.mean(illum) + eps)
    out = img / illum[None, None, :]
    return np.clip(out, 0.0, 1.0)


# ------------------------------
# Dataset
# ------------------------------
class DiamondTier7Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: Dict[str, int], train: bool, img_size: int):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.train = train

        t = []
        if train:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)))
            t.append(transforms.RandomHorizontalFlip(p=0.5))
            t.append(transforms.RandomRotation(8))
        else:
            t.append(transforms.Resize((img_size, img_size)))
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

        x = self.base_tf(img)  # [0,1]
        x_np = x.permute(1, 2, 0).numpy().astype(np.float32)
        x_np = shades_of_gray_wb(x_np, p=6)

        # Use full Lab if available
        if HAS_SKIMAGE:
            lab = skcolor.rgb2lab(x_np)
            L = lab[:, :, 0] / 100.0
            a = (lab[:, :, 1] + 128.0) / 255.0
            b = (lab[:, :, 2] + 128.0) / 255.0
            x_np = np.stack([L, a, b], axis=2).astype(np.float32)

        x = torch.from_numpy(x_np).permute(2, 0, 1)
        x = self.norm(x)
        return x, torch.tensor(y, dtype=torch.long)


# ------------------------------
# Ordinal (CORAL)
# ------------------------------
def build_coral_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    B = y.shape[0]
    K = num_classes - 1
    t = torch.zeros((B, K), device=y.device, dtype=torch.float32)
    for k in range(K):
        t[:, k] = (y > k).float()
    return t


class CoralLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
        targets = build_coral_targets(y, num_classes)
        return self.bce(logits, targets)


@torch.no_grad()
def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1).long()


# ------------------------------
# Metrics + confusion
# ------------------------------
@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, num_classes: int, use_ordinal: bool) -> Dict[str, object]:
    model.eval()
    total = 0
    correct = 0

    per_cls_total = np.zeros(num_classes, dtype=np.int64)
    per_cls_correct = np.zeros(num_classes, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        out = model(x)

        if use_ordinal:
            pred = coral_predict(out)
        else:
            pred = torch.argmax(out, dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

        y_np = y.detach().cpu().numpy()
        p_np = pred.detach().cpu().numpy()
        for yt, pt in zip(y_np, p_np):
            per_cls_total[yt] += 1
            per_cls_correct[yt] += int(pt == yt)
            cm[yt, pt] += 1

    acc = correct / max(1, total)
    per_cls_acc = per_cls_correct / np.maximum(1, per_cls_total)
    bacc = float(np.mean(per_cls_acc))
    return {
        "acc": float(acc),
        "bacc": float(bacc),
        "per_class_acc": per_cls_acc.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def save_json(path: Path, obj: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_eval_artifacts(folder: Path, report: Dict[str, object], labels: List[str]):
    """
    Writes:
      report.json
      confusion_matrix.csv  (rows=true, cols=pred, with labels)
      per_class_acc.csv
    """
    folder.mkdir(parents=True, exist_ok=True)

    # report.json
    save_json(folder / "report.json", report)

    # confusion_matrix.csv
    cm = np.array(report["confusion_matrix"], dtype=np.int64)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(folder / "confusion_matrix.csv", encoding="utf-8-sig")

    # per_class_acc.csv
    pca = np.array(report["per_class_acc"], dtype=np.float64)
    pca_df = pd.DataFrame({"label": labels, "per_class_acc": pca})
    pca_df.to_csv(folder / "per_class_acc.csv", index=False, encoding="utf-8-sig")


def write_best_and_history_eval_artifacts(
    out_dir: Path,
    report: Dict[str, object],
    labels: List[str],
    epoch: int,
    bacc: float,
):
    """
    Always updates:
      out_dir/best_eval/
    Also writes a unique history folder:
      out_dir/eval_history/e{epoch}_bacc{bacc}_{timestamp}/
    """
    # Always-updated best folder
    best_folder = out_dir / "best_eval"
    write_eval_artifacts(best_folder, report, labels)

    # History folder
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_bacc = f"{bacc*100:.2f}".replace(".", "p")
    hist_folder = out_dir / "eval_history" / f"e{epoch:02d}_bacc{safe_bacc}_{ts}"
    write_eval_artifacts(hist_folder, report, labels)

    log(f"✓ wrote eval artifacts -> {best_folder}")
    log(f"✓ wrote eval history   -> {hist_folder}")


# ------------------------------
# Model factory
# ------------------------------
def make_model(model_name: str, num_classes: int, use_ordinal: bool) -> nn.Module:
    out_dim = (num_classes - 1) if use_ordinal else num_classes

    if model_name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feats, out_dim))
        return base

    if model_name == "efficientnet_v2_s":
        base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_feats = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feats, out_dim))
        return base

    raise ValueError(f"Unknown model_name={model_name}")


# ------------------------------
# Main
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="Run evaluation only")
    ap.add_argument("--ckpt", type=str, default="", help="Checkpoint path for --eval")
    ap.add_argument("--epochs", type=int, default=CFG().epochs)
    ap.add_argument("--batch-size", type=int, default=CFG().batch_size)
    ap.add_argument("--img-size", type=int, default=CFG().img_size)
    ap.add_argument("--model", type=str, default=CFG().model_name, choices=["efficientnet_b0", "efficientnet_v2_s"])
    ap.add_argument("--no-ordinal", action="store_true", help="Disable ordinal (use normal CE)")
    ap.add_argument("--seed", type=int, default=CFG().seed)
    return ap.parse_args()


def main():
    global DEVICE
    args = parse_args()

    cfg = CFG()
    cfg.seed = int(args.seed)
    cfg.epochs = int(args.epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.img_size = int(args.img_size)
    cfg.model_name = str(args.model)
    cfg.use_ordinal = (not args.no_ordinal)

    seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("DIAMOND tier7 TRAIN (REAL-MAP OR ABORT)")
    log(f"DEVICE={DEVICE} | skimage={HAS_SKIMAGE}")
    log(f"OUT_DIR={out_dir}")
    log(f"MODEL={cfg.model_name} | ORDINAL={cfg.use_ordinal}")
    log("=" * 70)

    # 1) Download + locate CSV
    log("Downloading dataset via kagglehub...")
    t0 = time.time()
    ds_path = Path(kagglehub.dataset_download(cfg.dataset_handle))
    log(f"Download done in {time.time()-t0:.1f}s | Path: {ds_path}")

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
    img_index = build_image_indexes(imgs)

    # 3) Discover mapping column
    candidates = []
    for c in df.columns:
        if any(k in c for k in ["image", "img", "file", "path", "name", "id", "uuid"]):
            candidates.append(c)
    if not candidates:
        candidates = list(df.columns)

    log(f"Mapping candidates (ordered): {candidates[:25]}{' ...' if len(candidates)>25 else ''}")

    best = (0.0, None, None)
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
        diag = {
            "best_ratio": best_ratio,
            "best_col": best_col,
            "best_strategy": best_strat,
            "csv_columns": df.columns.tolist(),
            "num_images_found": len(imgs),
            "min_match_ratio_required": cfg.min_match_ratio,
        }
        save_json(out_dir / "mapping_diagnostics.json", diag)
        log(f"Saved diagnostics -> {out_dir/'mapping_diagnostics.json'}")
        return

    # 4) Materialize mapping
    df["image_path"] = materialize_mapping(df, best_col, best_strat, img_index, root_dir)
    before = len(df)
    df = df[df["image_path"].notna()].copy()
    df["image_path"] = df["image_path"].astype(str)
    after = len(df)
    log(f"[MAP] Kept mapped rows: {after}/{before} = {after/max(1,before):.2%}")

    # 5) Split (save for consistent eval)
    labels = TIER7_ORDER
    label2idx = {name: i for i, name in enumerate(labels)}
    df["y"] = df["tier7"].map(label2idx).astype(int)

    split_path = out_dir / "split_idx.json"
    if split_path.exists():
        sp = json.loads(split_path.read_text(encoding="utf-8"))
        train_idx = np.array(sp["train_idx"], dtype=np.int64)
        val_idx = np.array(sp["val_idx"], dtype=np.int64)
        log(f"[SPLIT] Loaded saved split -> {split_path}")
    else:
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
        train_idx = np.array(train_idx, dtype=np.int64)
        val_idx = np.array(val_idx, dtype=np.int64)
        save_json(split_path, {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(), "seed": cfg.seed})
        log(f"[SPLIT] Saved split -> {split_path}")

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    log(f"[SPLIT] Train={len(train_df)} | Val={len(val_df)} | Classes={len(labels)}")
    log("[TRAIN] tier7 distribution (train):")
    print(train_df["tier7"].value_counts(), flush=True)
    log("[VAL] tier7 distribution (val):")
    print(val_df["tier7"].value_counts(), flush=True)

    # 6) DataLoaders
    train_ds = DiamondTier7Dataset(train_df, label2idx, train=True, img_size=cfg.img_size)
    val_ds = DiamondTier7Dataset(val_df, label2idx, train=False, img_size=cfg.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # 7) Model
    model = make_model(cfg.model_name, num_classes=len(labels), use_ordinal=cfg.use_ordinal)
    model.to(DEVICE)

    # Freeze backbone initially
    if hasattr(model, "features"):
        for p in model.features.parameters():
            p.requires_grad = False

    # 8) Loss
    n_classes = len(labels)

    if cfg.use_ordinal:
        y_train = train_df["y"].values
        K = n_classes - 1
        pos = np.zeros(K, dtype=np.float64)
        neg = np.zeros(K, dtype=np.float64)
        for k in range(K):
            mask_pos = (y_train > k)
            pos[k] = mask_pos.sum()
            neg[k] = (~mask_pos).sum()
        pos_weight = (neg / np.maximum(1.0, pos)).astype(np.float32)
        pos_weight_t = torch.tensor(pos_weight, device=DEVICE, dtype=torch.float32)
        criterion = CoralLoss(pos_weight=pos_weight_t)
        log(f"[LOSS] CORAL pos_weight per threshold: {pos_weight.round(3).tolist()}")
    else:
        counts = train_df["y"].value_counts().sort_index()
        freq = np.array([counts.get(i, 0) for i in range(n_classes)], dtype=np.float64)
        cls_w = 1.0 / np.maximum(1.0, freq)
        cls_w = cls_w / cls_w.mean()
        class_weights = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
        log(f"[LOSS] CE class_weights (normalized): {cls_w.round(3).tolist()}")

    # Optimizer
    params = []
    head_lr = cfg.lr * 5.0
    if hasattr(model, "features") and hasattr(model, "classifier"):
        params.append({"params": model.features.parameters(), "lr": cfg.lr})
        params.append({"params": model.classifier.parameters(), "lr": head_lr})
    else:
        params.append({"params": model.parameters(), "lr": cfg.lr})

    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    # AMP
    use_amp = (DEVICE == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log(f"[AMP] enabled={use_amp}")

    # Paths
    best_path = out_dir / "best_effnet_tier7.pth"

    # Eval-only mode
    if args.eval:
        if not args.ckpt:
            raise ValueError("--eval requires --ckpt path")
        ckpt_path = Path(args.ckpt)
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.to(DEVICE)

        rep = eval_model(model, val_loader, num_classes=n_classes, use_ordinal=cfg.use_ordinal)
        rep["labels"] = labels
        rep["model"] = cfg.model_name
        rep["ordinal"] = cfg.use_ordinal
        rep["epoch"] = -1

        # Write artifacts to folders
        write_best_and_history_eval_artifacts(out_dir, rep, labels, epoch=0, bacc=rep["bacc"])

        log(f"[EVAL] acc={rep['acc']*100:.2f}% | bacc={rep['bacc']*100:.2f}%")
        return

    # 9) Train loop
    best_bacc = -1.0
    bad_epochs = 0

    log("=" * 70)
    log("START TRAIN")
    log("=" * 70)

    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.time()

        if epoch == cfg.unfreeze_at and hasattr(model, "features"):
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

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                if cfg.use_ordinal:
                    loss = criterion(out, y, n_classes)
                else:
                    loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

            if bi % 50 == 0 or bi == 1:
                avg_loss = running_loss / max(1, n_batches)
                log(f"  [train] batch {bi}/{len(train_loader)} | loss={avg_loss:.4f}")

        train_loss = running_loss / max(1, n_batches)
        rep = eval_model(model, val_loader, num_classes=n_classes, use_ordinal=cfg.use_ordinal)

        log(
            f"[VAL] acc={rep['acc']*100:.2f}% | bacc={rep['bacc']*100:.2f}% "
            f"| train_loss={train_loss:.4f} | epoch_time={time.time()-t_ep:.1f}s"
        )

        scheduler.step(rep["bacc"])

        # Save best
        if rep["bacc"] > best_bacc + 1e-4:
            best_bacc = rep["bacc"]
            bad_epochs = 0

            torch.save(model.state_dict(), best_path)

            rep_out = dict(rep)
            rep_out["labels"] = labels
            rep_out["best_bacc"] = float(best_bacc)
            rep_out["epoch"] = int(epoch)
            rep_out["model"] = cfg.model_name
            rep_out["ordinal"] = cfg.use_ordinal
            rep_out["train_loss"] = float(train_loss)

            # Write eval artifacts folders (best_eval + history)
            write_best_and_history_eval_artifacts(out_dir, rep_out, labels, epoch=epoch, bacc=best_bacc)

            log(f"✓ saved best checkpoint -> {best_path} | best_bacc={best_bacc*100:.2f}%")
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
    main()
