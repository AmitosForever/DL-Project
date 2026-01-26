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

        # Prefer SHAPE for this script
        if any("shape" in c for c in cols):
            s += 20

        # Need image/path column
        if any(k in c for c in cols for k in ["image", "img", "path", "file", "filename", "url"]):
            s += 10

        # weak signals (not the target)
        if any("cut" in c for c in cols):
            s += 1
        if any("polish" in c for c in cols):
            s += 1
        if any("symmetry" in c for c in cols):
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

    # Absolute path
    if os.path.isabs(s) and os.path.isfile(s):
        return s

    # Relative to root_dir
    p = os.path.join(root_dir, s.replace("/", os.sep))
    if os.path.isfile(p):
        return p

    # Filename lookup
    base = os.path.basename(s)
    base = re.split(r"[?#]", base)[0]
    if base in image_index:
        return image_index[base]

    return None


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
class ImageLabelDataset(Dataset):
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
# Model (ResNet-18)
# -----------------------------
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.0):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        if dropout and dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# -----------------------------
# AMP helpers (safe across versions)
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
# Transforms
# -----------------------------
def build_transforms(img_size: int, mean, std, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
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
class Metrics:
    loss: float
    acc: float
    bal_acc: float
    macro_f1: float


def metrics_from_confmat(conf: torch.Tensor) -> Tuple[float, float, float]:
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
    return acc, bal_acc, macro_f1


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
    label_smoothing: float,
    grad_clip: float,
    accum_steps: int,
) -> float:
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=max(0.0, float(label_smoothing)))

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

        with autocast_context(autocast_fn, device, enabled=use_amp):
            logits = model(x)
            loss = ce(logits, y)

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

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            batch_acc = (pred == y).float().mean().item()

        avg_loss = running / max(1, n)
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{batch_acc*100:.2f}%", lr=f"{lr_now:.2e}", upd=updates, accum=accum_steps)

    return running / max(1, n)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    desc: str = "val",
) -> Tuple[Metrics, torch.Tensor]:
    model.eval()
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

        logits = model(x)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        pred = logits.argmax(dim=1)
        idx = (y.to(torch.int64) * num_classes + pred.to(torch.int64))
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        conf += binc.view(num_classes, num_classes)

        acc, bal_acc, macro_f1 = metrics_from_confmat(conf)
        avg_loss = total_loss / max(1, n)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc*100:.2f}%", bal=f"{bal_acc*100:.2f}%")

    avg_loss = total_loss / max(1, n)
    acc, bal_acc, macro_f1 = metrics_from_confmat(conf)
    return Metrics(loss=avg_loss, acc=acc, bal_acc=bal_acc, macro_f1=macro_f1), conf


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
    ap.add_argument("--warmup_epochs", type=int, default=2)

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--img_size_ft", type=int, default=320)
    ap.add_argument("--ft_from_epoch", type=int, default=9)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--effective_batch", type=int, default=128)
    ap.add_argument("--accum_steps", type=int, default=None)

    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--out", default="shape_resnet18_best.pt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_decode_retries", type=int, default=10)

    ap.add_argument("--sampler", action="store_true", default=True)
    ap.add_argument("--no_sampler", action="store_true")

    ap.add_argument("--label_smoothing", type=float, default=0.02)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--per_class", action="store_true", default=True)
    ap.add_argument("--no_per_class", action="store_true")

    args = ap.parse_args()

    if args.no_sampler:
        args.sampler = False
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

    # IMPORTANT: SHAPE ONLY (unless user overrides with --label_col)
    label_col = args.label_col or guess_column(cols, ["shape"])

    if img_col is None:
        raise ValueError(f"לא הצלחתי לזהות עמודת תמונה. columns={cols}. תן --img_col ידנית.")
    if label_col is None:
        shape_like = [c for c in cols if "shape" in c.lower()]
        raise ValueError(
            f"לא מצאתי עמודת SHAPE. columns={cols}\n"
            f"עמודות שמכילות 'shape': {shape_like}\n"
            f"תן --label_col ידנית לעמודה הנכונה."
        )

    df = df.dropna(subset=[img_col, label_col]).copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    print(f"[INFO] img_col={img_col} label_col={label_col}")
    print("[INFO] label distribution (top 30):")
    print(df[label_col].value_counts().head(30))

    classes = sorted(df[label_col].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    num_classes = len(classes)

    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] classes={classes}")
    print(f"[INFO] GradAccum: batch={args.batch} effective_batch={args.effective_batch} accum_steps={args.accum_steps}")

    # Split
    train_idx, val_idx = stratified_split_indices(df[label_col].tolist(), args.val_split, args.seed)
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # Image index
    if args.root_dir is None:
        args.root_dir = os.path.dirname(args.csv)
    image_index = build_image_index(args.root_dir)

    # Transforms
    weights = ResNet18_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std
    tfm_train = build_transforms(args.img_size, mean, std, train=True)
    tfm_val = build_transforms(args.img_size_ft, mean, std, train=False)

    ds_train = ImageLabelDataset(
        df_train, img_col, label_col, args.root_dir, label2idx, image_index,
        tfm=tfm_train, max_decode_retries=args.max_decode_retries
    )
    ds_val = ImageLabelDataset(
        df_val, img_col, label_col, args.root_dir, label2idx, image_index,
        tfm=tfm_val, max_decode_retries=args.max_decode_retries
    )

    # Sampler
    sampler = None
    if args.sampler:
        y_train = df_train[label_col].tolist()
        counts: Dict[str, int] = {}
        for y in y_train:
            counts[y] = counts.get(y, 0) + 1
        sample_weights = [1.0 / counts[y] for y in y_train]
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

    model = ResNet18Classifier(num_classes=num_classes, dropout=args.dropout).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # AMP
    autocast_fn, scaler, amp_available = get_amp(device)
    use_amp = amp_available and (not args.no_amp)

    # Scheduler
    updates_per_epoch = int(math.ceil(len(dl_train) / max(1, args.accum_steps)))
    warmup_steps = updates_per_epoch * max(0, int(args.warmup_epochs))
    total_steps = updates_per_epoch * max(1, int(args.epochs))
    sched = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    sched.set_lr(0)

    best_bal = -1.0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.ft_from_epoch:
            print(f"[STAGE] Fine-tune from epoch {epoch}: img_size={args.img_size_ft}, lr={args.lr_ft}")
            ds_train.tfm = build_transforms(args.img_size_ft, mean, std, train=True)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr_ft

        train_loss = train_one_epoch(
            model=model,
            loader=dl_train,
            device=device,
            optimizer=optimizer,
            scheduler=sched,
            autocast_fn=autocast_fn,
            scaler=scaler,
            use_amp=use_amp,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            accum_steps=args.accum_steps,
        )

        val_metrics, conf = evaluate(model, dl_val, device, num_classes=num_classes, desc="val")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_acc={val_metrics.acc*100:.2f}% | "
            f"val_bal_acc={val_metrics.bal_acc*100:.2f}% | "
            f"val_macroF1={val_metrics.macro_f1:.4f}"
        )

        if args.per_class:
            print_per_class(conf, classes, title=f"[VAL] Per-class @ epoch {epoch:02d}")

        if val_metrics.bal_acc > best_bal:
            best_bal = val_metrics.bal_acc
            ckpt = {
                "model": model.state_dict(),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "classes": classes,
                "img_col": img_col,
                "label_col": label_col,
                "root_dir": args.root_dir,
                "backbone": "resnet18",
                "weights": "ResNet18_Weights.DEFAULT",
                "task": "shape_classification",
                "train": {
                    "img_size": args.img_size,
                    "img_size_ft": args.img_size_ft,
                    "ft_from_epoch": args.ft_from_epoch,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "lr_ft": args.lr_ft,
                    "wd": args.wd,
                    "batch": args.batch,
                    "effective_batch": args.effective_batch,
                    "accum_steps": args.accum_steps,
                    "label_smoothing": args.label_smoothing,
                    "grad_clip": args.grad_clip,
                    "sampler": args.sampler,
                }
            }
            torch.save(ckpt, args.out)
            print(f"[SAVE] best -> {args.out} (val_bal_acc={best_bal*100:.2f}%)")

    print(f"[DONE] best_val_bal_acc={best_bal*100:.2f}% | saved={args.out}")


if __name__ == "__main__":
    main()
