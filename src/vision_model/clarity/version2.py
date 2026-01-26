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


def guess_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols = [c.lower() for c in columns]
    for cand in candidates:
        for i, c in enumerate(cols):
            if cand in c:
                return columns[i]
    return None


def normalize_clarity_label(s: str, merge_si3_to_si2: bool = True) -> str:
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
# Crop cache
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

    edges = cv2.Canny(gray, 50, 150)
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


def preprocess_crops(df: pd.DataFrame, img_col: str, root_dir: Optional[str], cache_dir: str) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)

    has_cv2 = False
    try:
        import cv2  # noqa
        has_cv2 = True
    except Exception:
        has_cv2 = False

    new_paths = []
    pbar = tqdm(range(len(df)), desc="preprocess_crops", leave=True)
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
            new_paths.append(rel)

        if (i + 1) % 500 == 0:
            pbar.set_postfix(bad=bad)

    df2 = df.copy()
    df2[img_col] = new_paths
    return df2


# -----------------------------
# Dataset
# -----------------------------
class ClarityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, label_col: str,
                 root_dir: Optional[str], label2idx: Dict[str, int], tfm=None,
                 max_decode_retries: int = 10):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.label_col = label_col
        self.root_dir = root_dir
        self.label2idx = label2idx
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

            img_path = resolve_path(p, self.root_dir)
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
# Model
# -----------------------------
class ClarityNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)

        in_features = self.backbone.classifier[2].in_features  # usually 768
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# -----------------------------
# AMP
# -----------------------------
def get_amp(device: torch.device):
    if device.type != "cuda":
        return None, None, False
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.autocast, torch.amp.GradScaler("cuda"), True
    return torch.cuda.amp.autocast, torch.cuda.amp.GradScaler(), True


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


def apply_mixup_cutmix(x, y, num_classes: int, mixup_alpha: float, cutmix_alpha: float, p_mix: float):
    y1 = one_hot(y, num_classes)

    if p_mix <= 0 or torch.rand(1).item() > p_mix:
        return x, y1

    r = torch.rand(1).item()
    if r < 0.5:
        if mixup_alpha <= 0:
            return x, y1
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        perm = torch.randperm(x.size(0), device=x.device)
        x2 = x[perm]
        y2 = y1[perm]
        return lam * x + (1 - lam) * x2, lam * y1 + (1 - lam) * y2
    else:
        if cutmix_alpha <= 0:
            return x, y1
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
        return x_aug, lam_adj * y1 + (1 - lam_adj) * y2


def soft_cross_entropy(logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(y_soft * logp).sum(dim=1).mean()


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


def print_per_class_report(conf: torch.Tensor, classes: List[str], title: str):
    conf = conf.detach().cpu().to(torch.float32)
    tp = torch.diag(conf)
    support = conf.sum(dim=1).clamp_min(1.0)
    pred_sum = conf.sum(dim=0).clamp_min(1.0)

    acc_c = tp / support
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


@torch.no_grad()
def evaluate(model, loader, device, criterion_hard, tta: bool = False) -> Tuple[Metrics, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    n = 0
    num_classes = model.backbone.classifier[-1].out_features
    conf = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if not tta:
            logits = model(x)
        else:
            logits1 = model(x)
            logits2 = model(torch.flip(x, dims=[3]))
            logits = (logits1 + logits2) / 2.0

        loss = criterion_hard(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        pred = logits.argmax(dim=1)
        for t, p in zip(y, pred):
            conf[t.long(), p.long()] += 1

    avg_loss = total_loss / max(n, 1)
    acc, macro_f1 = metrics_from_confmat(conf)
    return Metrics(loss=avg_loss, acc=acc, macro_f1=macro_f1), conf


# -----------------------------
# Manual warmup+cosine LR
# -----------------------------
def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step <= warmup_steps:
        return step / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, base_lr: float, factor: float):
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * factor


# -----------------------------
# Train
# -----------------------------
def train_one_epoch(
    model, loader, device, optimizer, autocast, scaler, use_amp: bool,
    num_classes: int, mix_p: float, mixup_alpha: float, cutmix_alpha: float,
    log_every: int, base_lr: float, warmup_steps: int, total_steps: int, global_step: int
) -> Tuple[float, int]:
    model.train()
    running = 0.0
    n = 0
    t0 = time.time()

    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc="train", leave=False)
    for step_in_epoch, (x, y) in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        x_aug, y_soft = apply_mixup_cutmix(
            x, y, num_classes=num_classes,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, p_mix=mix_p
        )

        if use_amp:
            with autocast("cuda"):
                logits = model(x_aug)
                loss = soft_cross_entropy(logits, y_soft)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_aug)
            loss = soft_cross_entropy(logits, y_soft)
            loss.backward()
            optimizer.step()

        # update LR AFTER optimizer step
        global_step += 1
        factor = lr_factor(global_step, warmup_steps, total_steps)
        set_lr(optimizer, base_lr, factor)

        running += loss.item() * x.size(0)
        n += x.size(0)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            batch_acc = (pred == y).float().mean().item()

        if step_in_epoch % log_every == 0 or step_in_epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = running / max(n, 1)
            elapsed = time.time() - t0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", batch_acc=f"{batch_acc*100:.2f}%", lr=f"{lr:.2e}", t=f"{elapsed:.0f}s")

    return running / max(n, 1), global_step


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default=None)
    ap.add_argument("--root_dir", default=None)
    ap.add_argument("--img_col", default=None)
    ap.add_argument("--label_col", default=None)

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out", default="clarity_best.pt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_decode_retries", type=int, default=10)

    ap.add_argument("--normalize_labels", action="store_true", default=True)
    ap.add_argument("--no_normalize_labels", action="store_true")
    ap.add_argument("--merge_si3_to_si2", action="store_true", default=True)

    ap.add_argument("--crop_cache", action="store_true", default=True)
    ap.add_argument("--no_crop_cache", action="store_true")
    ap.add_argument("--cache_dir", default="_clarity_cache_crops")

    ap.add_argument("--no_sampler", action="store_true")

    ap.add_argument("--mix_p", type=float, default=0.8)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--cutmix_alpha", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--per_class", action="store_true", default=True)
    ap.add_argument("--no_per_class", action="store_true")

    args = ap.parse_args()

    if args.no_normalize_labels:
        args.normalize_labels = False
    if args.no_crop_cache:
        args.crop_cache = False
    if args.no_per_class:
        args.per_class = False

    seed_all(args.seed)

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

    if args.crop_cache:
        cache_root = os.path.join(os.path.dirname(args.csv), args.cache_dir)
        print(f"[INFO] Crop cache: ON -> {cache_root}")
        df = preprocess_crops(df, img_col=img_col, root_dir=args.root_dir, cache_dir=cache_root)
        args.root_dir = cache_root
    else:
        print("[INFO] Crop cache: OFF")

    classes = sorted(df[label_col].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    num_classes = len(classes)

    print(f"[INFO] img_col={img_col} label_col={label_col} num_classes={num_classes}")
    print(f"[INFO] classes={classes}")
    print("[INFO] BACKBONE=ConvNeXt-Tiny | weights=ConvNeXt_Tiny_Weights.DEFAULT (ImageNet)")
    print(f"[INFO] IMG_SIZE={args.img_size}x{args.img_size} | batch={args.batch}")

    train_idx, val_idx = stratified_split_indices(df[label_col].tolist(), args.val_split, args.seed)
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    weights = ConvNeXt_Tiny_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std

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

    ds_train = ClarityDataset(df_train, img_col, label_col, args.root_dir, label2idx, tfm=train_tfm,
                              max_decode_retries=args.max_decode_retries)
    ds_val = ClarityDataset(df_val, img_col, label_col, args.root_dir, label2idx, tfm=val_tfm,
                            max_decode_retries=args.max_decode_retries)

    if args.no_sampler:
        sampler = None
        print("[INFO] Sampler: OFF")
    else:
        y_train = df_train[label_col].tolist()
        counts: Dict[str, int] = {}
        for y in y_train:
            counts[y] = counts.get(y, 0) + 1
        sample_weights = [1.0 / counts[y] for y in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print("[INFO] Sampler: WeightedRandomSampler ON")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClarityNet(num_classes=num_classes, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # init LR at 0 for warmup
    set_lr(optimizer, args.lr, 0.0)

    steps_per_epoch = len(dl_train)
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)
    total_steps = max(1, args.epochs * steps_per_epoch)

    autocast, scaler, amp_available = get_amp(device)
    use_amp = amp_available and (not args.no_amp)

    criterion_hard = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_acc = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, dl_train, device, optimizer,
            autocast=autocast, scaler=scaler, use_amp=use_amp,
            num_classes=num_classes,
            mix_p=args.mix_p, mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
            log_every=args.log_every,
            base_lr=args.lr, warmup_steps=warmup_steps, total_steps=total_steps, global_step=global_step
        )

        val_metrics, conf = evaluate(model, dl_val, device, criterion_hard, tta=False)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_acc={val_metrics.acc*100:.2f}% | "
            f"val_macroF1={val_metrics.macro_f1:.4f}"
        )

        if args.per_class:
            print_per_class_report(conf, classes, title=f"[VAL] Per-class @ epoch {epoch:02d}")

        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
            best_acc = val_metrics.acc
            ckpt = {
                "model": model.state_dict(),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "img_col": img_col,
                "label_col": label_col,
                "root_dir": args.root_dir,
                "img_size": args.img_size,
                "classes": classes,
                "backbone": "convnext_tiny",
                "weights": "ConvNeXt_Tiny_Weights.DEFAULT",
                "normalize_labels": bool(args.normalize_labels),
                "crop_cache": bool(args.crop_cache),
                "sampler": (sampler is not None),
                "mixup_cutmix": {"p": args.mix_p, "mixup_alpha": args.mixup_alpha, "cutmix_alpha": args.cutmix_alpha},
            }
            torch.save(ckpt, args.out)
            print(f"[SAVE] best -> {args.out} (macroF1={best_f1:.4f}, acc={best_acc*100:.2f}%)")

    tta_metrics, tta_conf = evaluate(model, dl_val, device, criterion_hard, tta=True)
    print(
        f"[TTA] val_loss={tta_metrics.loss:.4f} | "
        f"val_acc={tta_metrics.acc*100:.2f}% | "
        f"val_macroF1={tta_metrics.macro_f1:.4f}"
    )
    if args.per_class:
        print_per_class_report(tta_conf, classes, title="[TTA] Per-class on VAL (final)")

    print(f"[DONE] best_macroF1={best_f1:.4f} | best_acc={best_acc*100:.2f}% | saved={args.out}")


if __name__ == "__main__":
    main()
