import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from PIL import Image, ImageFile

# Allow loading truncated/corrupt-ish JPEGs instead of crashing immediately
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Utilities
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


def guess_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols = [c.lower() for c in columns]
    for cand in candidates:
        for i, c in enumerate(cols):
            if cand in c:
                return columns[i]
    return None


def macro_f1_from_confmat(conf: torch.Tensor) -> float:
    tp = torch.diag(conf).float()
    fp = conf.sum(dim=0).float() - tp
    fn = conf.sum(dim=1).float() - tp

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    support = conf.sum(dim=1).float()
    valid = support > 0
    if valid.any():
        return f1[valid].mean().item()
    return f1.mean().item()


def normalize_clarity_label(s: str) -> str:
    """
    Normalize noisy clarity labels from web-scraped dataset.
    - SI2- -> SI2
    - I4..I9 -> I3 (standard scale is I1-I3)
    """
    if s is None:
        return ""
    s = str(s).strip().upper()

    if s == "SI2-":
        return "SI2"

    if s.startswith("I") and len(s) == 2 and s[1].isdigit():
        try:
            v = int(s[1])
            if v >= 4:
                return "I3"
        except Exception:
            pass

    return s


def stratified_split_indices(labels: List[str], val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    """
    Simple stratified split by label.
    For labels with count==1, keep it in train.
    """
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
        n_val = min(n_val, n - 1)  # leave at least 1 for train
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# -----------------------------
# Kaggle auto download (optional)
# -----------------------------
def auto_download_diamond_dataset() -> str:
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            "kagglehub לא מותקן. התקן: pip install kagglehub\n"
            "או תן --csv ידנית."
        ) from e

    return kagglehub.dataset_download("aayushpurswani/diamond-images-dataset")


def find_best_csv(ds_root: str) -> str:
    csvs = []
    for r, _, files in os.walk(ds_root):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(r, f))

    if not csvs:
        raise FileNotFoundError(f"לא נמצאו קבצי CSV תחת: {ds_root}")

    def score_csv(p: str) -> int:
        try:
            df = pd.read_csv(p, nrows=50)
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
# Dataset
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
    ):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.root_dir = root_dir
        self.label2idx = label2idx
        self.tfm = tfm
        self.max_decode_retries = max_decode_retries

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, p: str) -> str:
        p = str(p).replace("/", os.sep)
        if self.root_dir and not os.path.isabs(p):
            return os.path.join(self.root_dir, p)
        return p

    def _load_image(self, img_path: str) -> Image.Image:
        img = Image.open(img_path)
        img = img.convert("RGB")
        return img

    def __getitem__(self, idx):
        last_err = None

        for _ in range(self.max_decode_retries):
            row = self.df.iloc[idx]
            img_path = self._resolve_path(row[self.img_col])
            y_str = str(row[self.label_col]).strip()

            if y_str not in self.label2idx:
                idx = torch.randint(0, len(self.df), (1,)).item()
                continue

            try:
                img = self._load_image(img_path)
                if self.tfm is not None:
                    img = self.tfm(img)
                y = self.label2idx[y_str]
                return img, y
            except Exception as e:
                last_err = (img_path, repr(e))
                idx = torch.randint(0, len(self.df), (1,)).item()

        raise RuntimeError(
            f"Failed to decode images after {self.max_decode_retries} retries. "
            f"Last error on file: {last_err[0]} | {last_err[1]}"
        )


# -----------------------------
# Model (Backbone)
# -----------------------------
class ClarityNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        # BACKBONE: EfficientNetV2-S pretrained on ImageNet
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = efficientnet_v2_s(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# -----------------------------
# AMP compatibility (new/old)
# -----------------------------
def get_amp(device: torch.device):
    if device.type != "cuda":
        return None, None, False

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.autocast, torch.amp.GradScaler("cuda"), True

    return torch.cuda.amp.autocast, torch.cuda.amp.GradScaler(), True


# -----------------------------
# Train / Eval
# -----------------------------
@dataclass
class Metrics:
    loss: float
    acc: float
    macro_f1: float


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Metrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    num_classes = model.backbone.classifier[-1].out_features
    conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)

        for t, p in zip(y, pred):
            conf[t.long(), p.long()] += 1

    avg_loss = total_loss / max(n, 1)
    acc = correct / max(n, 1)
    macro_f1 = macro_f1_from_confmat(conf)

    return Metrics(loss=avg_loss, acc=acc, macro_f1=macro_f1)


def train_one_epoch(model, loader, device, criterion, optimizer, autocast, scaler, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / max(n, 1)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to CSV (auto-download if omitted)")
    ap.add_argument("--root_dir", default=None, help="Optional root dir to prepend to relative image paths")
    ap.add_argument("--img_col", default=None, help="Column name for image path (auto-detect if omitted)")
    ap.add_argument("--label_col", default=None, help="Column name for clarity label (auto-detect if omitted)")

    ap.add_argument("--epochs", type=int, default=15)

    # For 512x512, default batch is smaller to avoid OOM
    ap.add_argument("--batch", type=int, default=8)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)

    # >>> DEFAULT 512x512 <<<
    ap.add_argument("--img_size", type=int, default=512)

    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=1)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out", default="clarity_best.pt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_decode_retries", type=int, default=10)

    # data stability knobs
    ap.add_argument("--normalize_labels", action="store_true",
                    help="Normalize clarity labels (recommended for this dataset).")
    ap.add_argument("--no_class_weights", action="store_true",
                    help="Disable class weights (useful if weights make loss unstable).")
    ap.add_argument("--max_class_weight", type=float, default=20.0,
                    help="Cap for class weights to avoid extreme loss spikes (default=20).")

    args = ap.parse_args()

    seed_all(args.seed)

    # Auto CSV download if user didn't provide --csv
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
        raise ValueError(
            f"לא הצלחתי לזהות עמודות אוטומטית.\n"
            f"עמודות שנמצאו: {cols}\n"
            f"תן --img_col ו--label_col ידנית."
        )

    df = df.dropna(subset=[img_col, label_col]).copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    if args.normalize_labels:
        df[label_col] = df[label_col].apply(normalize_clarity_label)

    classes = sorted(df[label_col].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    num_classes = len(classes)

    print(f"[INFO] img_col={img_col} label_col={label_col} num_classes={num_classes}")
    print(f"[INFO] classes={classes}")
    print("[INFO] BACKBONE=EfficientNetV2-S | weights=EfficientNet_V2_S_Weights.DEFAULT (ImageNet)")
    print(f"[INFO] IMG_SIZE={args.img_size}x{args.img_size} | batch={args.batch}")

    # Stratified split by label
    labels_list = df[label_col].tolist()
    train_idx, val_idx = stratified_split_indices(labels_list, args.val_split, args.seed)
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # Transforms
    weights = EfficientNet_V2_S_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std

    # For clarity: avoid aggressive zoom-out crops that remove tiny details
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.90, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    ds_train = ClarityDataset(
        df_train, img_col, label_col, args.root_dir, label2idx,
        tfm=train_tfm, max_decode_retries=args.max_decode_retries
    )
    ds_val = ClarityDataset(
        df_val, img_col, label_col, args.root_dir, label2idx,
        tfm=val_tfm, max_decode_retries=args.max_decode_retries
    )

    dl_train = DataLoader(
        ds_train, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClarityNet(num_classes=num_classes, dropout=args.dropout).to(device)

    # Class weights (gentle + capped)
    if args.no_class_weights:
        w = None
        print("[INFO] Class weights: OFF")
    else:
        counts = torch.zeros(num_classes, dtype=torch.long)
        for v in df_train[label_col].tolist():
            v = str(v).strip()
            if v in label2idx:
                counts[label2idx[v]] += 1

        freq = counts.float() / counts.sum().clamp_min(1)
        w = 1.0 / torch.sqrt(freq + 1e-12)  # gentler than full inverse
        w = w / w.mean().clamp_min(1e-12)
        w = torch.clamp(w, max=args.max_class_weight)
        w = w.to(device)
        print("[INFO] Class weights: ON (sqrt inverse freq + capped)")

    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)

    # Freeze backbone initially
    def set_backbone_trainable(trainable: bool):
        for p in model.backbone.features.parameters():
            p.requires_grad = trainable

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    autocast, scaler, amp_available = get_amp(device)
    use_amp = amp_available and (not args.no_amp)

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.freeze_backbone_epochs:
            set_backbone_trainable(False)
        else:
            set_backbone_trainable(True)

        train_loss = train_one_epoch(model, dl_train, device, criterion, optimizer, autocast, scaler, use_amp)
        val_metrics = evaluate(model, dl_val, device, criterion)
        scheduler.step()

        # val accuracy printed as percent
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_acc={val_metrics.acc*100:.2f}% | "
            f"val_macroF1={val_metrics.macro_f1:.4f}"
        )

        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
            best_state = {
                "model": model.state_dict(),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "img_col": img_col,
                "label_col": label_col,
                "root_dir": args.root_dir,
                "img_size": args.img_size,
                "classes": classes,
                "backbone": "efficientnet_v2_s",
                "weights": "EfficientNet_V2_S_Weights.DEFAULT",
                "normalize_labels": bool(args.normalize_labels),
            }
            torch.save(best_state, args.out)
            print(f"[SAVE] best checkpoint -> {args.out} (macroF1={best_f1:.4f})")

    print(f"[DONE] best_macroF1={best_f1:.4f} saved={args.out}")


if __name__ == "__main__":
    main()
