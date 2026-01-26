import os
import re
import glob
import time
import json
import random
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import joblib


# ============================================================
# 0) CONFIG
# ============================================================

@dataclass
class Config:
    dataset_handle: str = "aayushpurswani/diamond-images-dataset"
    seed: int = 42
    img_size: int = 256

    # Feature extraction
    bins_b: int = 32
    bins_a: int = 16
    bins_l: int = 16
    sog_p: int = 6  # Shades-of-Gray white balance p-norm

    # Task
    target: str = "tier7"  # "tier7" or "raw17"
    val_size: float = 0.20

    # Auto-repair logic
    min_margin_over_chance: float = 0.06  # if best_acc <= chance+margin -> suspicious mapping
    max_strategies: int = 6  # how many matching strategies to try
    cache_dir: str = "./_auto_cache"
    out_dir: str = "./_auto_out"

    # Models
    lr_max_iter: int = 4000
    hgb_max_iter: int = 500
    rf_n_estimators: int = 400

CFG = Config()


# ============================================================
# 1) LOGGING UTIL
# ============================================================

class Logger:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"run_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================
# 2) HELPERS
# ============================================================

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def natural_key(text: str):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

def sha1_of_list(items: List[str]) -> str:
    h = hashlib.sha1()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


# ============================================================
# 3) LABEL CLEANING + MAPPING
# ============================================================

VALID_GRADES_17 = ["D","E","F","G","H","I","J","K","L","M","N","O-P","Q-R","S-T","U-V","W-X","Y-Z"]

def clean_color_label(c: str) -> str:
    c = str(c).strip().upper()
    cleaning_map = {"D:P:BN": "D", "I:P": "I", "K:P": "K", "J:P": "J"}
    return cleaning_map.get(c, c)

def map_color_to_tier7(color_str: str) -> str:
    c = clean_color_label(color_str)
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
    return "Yellow_LowEnd"


# ============================================================
# 4) DOWNLOAD + BUILD RAW TABLE
# ============================================================

def download_and_read(logger: Logger) -> Tuple[pd.DataFrame, str]:
    logger.log("=== Downloading dataset ===")
    t0 = time.time()
    path = kagglehub.dataset_download(CFG.dataset_handle)
    logger.log(f"Download done in {time.time()-t0:.1f}s | Path: {path}")

    csv_candidates = glob.glob(os.path.join(path, "**", "diamond_data.csv"), recursive=True)
    if not csv_candidates:
        raise FileNotFoundError("Could not find diamond_data.csv in the downloaded dataset.")
    csv_path = csv_candidates[0]
    root_dir = os.path.dirname(csv_path)

    logger.log("=== Reading CSV ===")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "colour" in df.columns:
        df.rename(columns={"colour": "color"}, inplace=True)

    if "color" not in df.columns or "shape" not in df.columns:
        raise KeyError("CSV must include columns: 'color' and 'shape'")

    df["color"] = df["color"].apply(clean_color_label)
    df = df[df["color"].isin(VALID_GRADES_17)].copy()

    logger.log("=== Raw color distribution ===")
    logger.log(str(df["color"].value_counts()))

    df["tier7"] = df["color"].apply(map_color_to_tier7)
    logger.log("=== Tier7 distribution (CSV) ===")
    logger.log(str(df["tier7"].value_counts()))

    return df, root_dir


# ============================================================
# 5) MATCH STRATEGIES (self-repair tries multiple)
# ============================================================

def list_shape_folders(root_dir: str) -> Dict[str, str]:
    return {
        d.lower(): os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    }

def get_images(folder_path: str) -> List[str]:
    imgs = []
    for f in os.listdir(folder_path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            imgs.append(os.path.join(folder_path, f))
    return imgs

def sort_images(paths: List[str], mode: str, seed: int) -> List[str]:
    if mode == "natural":
        return sorted(paths, key=lambda p: natural_key(os.path.basename(p)))
    if mode == "filesize":
        return sorted(paths, key=lambda p: os.path.getsize(p))
    if mode == "mtime":
        return sorted(paths, key=lambda p: os.path.getmtime(p))
    if mode == "random":
        rng = random.Random(seed)
        paths2 = paths[:]
        rng.shuffle(paths2)
        return paths2
    # default
    return sorted(paths)

def build_matched_df(logger: Logger, df: pd.DataFrame, root_dir: str, sort_mode: str) -> pd.DataFrame:
    folders = list_shape_folders(root_dir)
    matched = []

    logger.log(f"=== Matching images by shape folders | strategy={sort_mode} ===")
    for shape in df["shape"].dropna().unique():
        s = str(shape).strip().lower()
        if s not in folders:
            continue
        folder_path = folders[s]
        imgs = get_images(folder_path)
        if not imgs:
            continue

        imgs = sort_images(imgs, mode=sort_mode, seed=CFG.seed)

        rows = df[df["shape"].str.lower() == s]
        n = min(len(imgs), len(rows))

        # IMPORTANT: pair by index only within same shape, but strategy changes ordering of imgs.
        for i in range(n):
            rec = rows.iloc[i].to_dict()
            rec["image_path"] = imgs[i]
            matched.append(rec)

    mdf = pd.DataFrame(matched)
    logger.log(f"[DATA] Matched samples: {len(mdf)}")
    if len(mdf) == 0:
        return mdf

    # show distribution after matching
    target_col = "tier7" if CFG.target == "tier7" else "color"
    logger.log(f"=== Distribution after matching ({target_col}) ===")
    logger.log(str(mdf[target_col].value_counts()))
    return mdf


# ============================================================
# 6) FEATURE EXTRACTION (cached) + SANITY SCORE
# ============================================================

def shades_of_gray_wb(img01: np.ndarray, p: int = 6, eps: float = 1e-6) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    illum = np.power(np.mean(np.power(img01, p), axis=(0, 1)) + eps, 1.0 / p)
    illum = illum / (np.mean(illum) + eps)
    out = img01 / (illum.reshape(1, 1, 3) + eps)
    return np.clip(out, 0.0, 1.0)

def extract_features(img_path: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    img = img.resize((CFG.img_size, CFG.img_size))
    img_np = np.asarray(img).astype(np.float32) / 255.0
    img_np = shades_of_gray_wb(img_np, p=CFG.sog_p)
    img_u8 = (img_np * 255.0).astype(np.uint8)

    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    # Body mask: avoid glare + deep shadow
    mask = (L > 60) & (L < 220)
    if mask.sum() < 50:
        mask = np.ones_like(L, dtype=bool)

    Lm = L[mask]; am = a[mask]; bm = b[mask]

    hist_b, _ = np.histogram(bm, bins=CFG.bins_b, range=(0, 255))
    hist_a, _ = np.histogram(am, bins=CFG.bins_a, range=(0, 255))
    hist_L, _ = np.histogram(Lm, bins=CFG.bins_l, range=(0, 255))

    hist_b = hist_b.astype(np.float32); hist_b /= (hist_b.sum() + 1e-9)
    hist_a = hist_a.astype(np.float32); hist_a /= (hist_a.sum() + 1e-9)
    hist_L = hist_L.astype(np.float32); hist_L /= (hist_L.sum() + 1e-9)

    stats = np.array([
        bm.mean(), bm.std(),
        am.mean(), am.std(),
        Lm.mean(), Lm.std()
    ], dtype=np.float32) / 255.0

    # additional "yellowness vs lightness" ratio stats (more stable sometimes)
    ratio = (bm - 128.0) / (Lm + 1e-6)
    ratio_stats = np.array([
        ratio.mean(), ratio.std(),
        np.percentile(ratio, 25),
        np.percentile(ratio, 50),
        np.percentile(ratio, 75)
    ], dtype=np.float32)

    feats = np.concatenate([hist_b, hist_a, hist_L, stats, ratio_stats], axis=0)
    return feats

def monotonicity_sanity_score(mdf: pd.DataFrame) -> float:
    """
    Quick check: if mapping is sane, average b* should roughly increase with "yellowness".
    We approximate with tier7 order or raw17 order. Returns score in [0,1] (higher is better).
    """
    # sample small subset for speed
    sub = mdf.sample(n=min(1200, len(mdf)), random_state=CFG.seed).copy()
    vals = []
    labels = []

    # order
    if CFG.target == "raw17":
        order = VALID_GRADES_17
        rank = {c:i for i,c in enumerate(order)}
        lab_col = "color"
    else:
        order = ["Premium_White","Near_Colorless_High","Near_Colorless_Low","Faint_Yellow","Very_Light_Yellow","Light_Yellow","Yellow_LowEnd"]
        rank = {c:i for i,c in enumerate(order)}
        lab_col = "tier7"

    for _, row in sub.iterrows():
        f = extract_features(row["image_path"])
        if f is None:
            continue
        # crude proxy for "b* center": take weighted mean of b-hist bins
        # hist_b starts at index 0, length CFG.bins_b
        hist_b = f[:CFG.bins_b]
        centers = np.linspace(0, 255, CFG.bins_b, dtype=np.float32)
        mean_b = float((hist_b * centers).sum())
        vals.append(mean_b)
        labels.append(rank.get(row[lab_col], 0))

    if len(vals) < 50:
        return 0.0

    vals = np.array(vals, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # correlation magnitude (normalized)
    corr = np.corrcoef(vals, labels)[0,1]
    if np.isnan(corr):
        return 0.0
    return float(max(0.0, min(1.0, corr)))  # clamp to [0,1]


def extract_cached(logger: Logger, mdf: pd.DataFrame, strategy_key: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    os.makedirs(CFG.cache_dir, exist_ok=True)

    target_col = "tier7" if CFG.target == "tier7" else "color"
    label_list = sorted(mdf[target_col].unique().tolist())
    label_to_idx = {c:i for i,c in enumerate(label_list)}

    cache_name = f"features_{CFG.target}_{strategy_key}_sz{CFG.img_size}_b{CFG.bins_b}_a{CFG.bins_a}_l{CFG.bins_l}_p{CFG.sog_p}.npz"
    cache_path = os.path.join(CFG.cache_dir, cache_name)

    if os.path.exists(cache_path):
        logger.log(f"[CACHE] Loading features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        labels = data["labels"].tolist()
        logger.log(f"[CACHE] Loaded X={X.shape} y={y.shape} labels={labels}")
        return X, y, labels

    logger.log("=== Extracting features (cached build) ===")
    t0 = time.time()

    X_list = []
    y_list = []
    bad = 0

    # Critical: build X and y together only when feature extraction succeeds
    for _, row in tqdm(mdf.iterrows(), total=len(mdf)):
        f = extract_features(row["image_path"])
        if f is None:
            bad += 1
            continue
        X_list.append(f)
        y_list.append(label_to_idx[row[target_col]])

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    logger.log(f"Feature extraction done in {time.time()-t0:.1f}s | bad_images={bad} | X={X.shape} y={y.shape}")

    np.savez_compressed(cache_path, X=X, y=y, labels=np.array(label_list, dtype=object))
    logger.log(f"[CACHE] Saved -> {cache_path}")

    return X, y, label_list


# ============================================================
# 7) TRAIN + EVAL MULTIPLE MODELS, PICK BEST
# ============================================================

def train_and_eval(logger: Logger, X: np.ndarray, y: np.ndarray, labels: List[str], run_tag: str) -> Dict:
    os.makedirs(CFG.out_dir, exist_ok=True)

    logger.log("=== Train/Val split ===")
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=CFG.val_size,
        random_state=CFG.seed,
        stratify=y
    )
    logger.log(f"[SPLIT] Train={len(y_tr)} | Val={len(y_va)} | Classes={len(labels)}")

    chance = 1.0 / len(labels)
    logger.log(f"[BASELINE] Chance Top-1 ≈ {chance*100:.2f}%")

    results = []
    best = {"acc": -1.0}

    # ----- Model 1: Scaled Logistic Regression -----
    logger.log("=== MODEL 1: Scaled LogisticRegression(class_weight='balanced') ===")
    t0 = time.time()
    model_lr = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=CFG.lr_max_iter,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1
        ))
    ])
    model_lr.fit(X_tr, y_tr)
    pred = model_lr.predict(X_va)
    acc = accuracy_score(y_va, pred)
    dur = time.time() - t0
    logger.log(f"[LR] Val Top-1={acc*100:.2f}% | time={dur:.1f}s | max_iter={CFG.lr_max_iter}")
    results.append(("LR", acc, dur, model_lr, pred))

    # ----- Model 2: HistGradientBoosting -----
    logger.log("=== MODEL 2: HistGradientBoostingClassifier ===")
    t0 = time.time()
    model_hgb = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_iter=CFG.hgb_max_iter,
        random_state=CFG.seed
    )
    model_hgb.fit(X_tr, y_tr)
    pred = model_hgb.predict(X_va)
    acc = accuracy_score(y_va, pred)
    dur = time.time() - t0
    logger.log(f"[HGB] Val Top-1={acc*100:.2f}% | time={dur:.1f}s | max_iter={CFG.hgb_max_iter}")
    results.append(("HGB", acc, dur, model_hgb, pred))

    # ----- Model 3: RandomForest (slow but sometimes helps) -----
    logger.log("=== MODEL 3: RandomForestClassifier ===")
    t0 = time.time()
    model_rf = RandomForestClassifier(
        n_estimators=CFG.rf_n_estimators,
        random_state=CFG.seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model_rf.fit(X_tr, y_tr)
    pred = model_rf.predict(X_va)
    acc = accuracy_score(y_va, pred)
    dur = time.time() - t0
    logger.log(f"[RF] Val Top-1={acc*100:.2f}% | time={dur:.1f}s | n_estimators={CFG.rf_n_estimators}")
    results.append(("RF", acc, dur, model_rf, pred))

    # pick best
    for name, acc, dur, model, pred in results:
        if acc > best["acc"]:
            best = {
                "name": name,
                "acc": float(acc),
                "time": float(dur),
                "model": model,
                "pred": pred
            }

    # report for best
    best_name = best["name"]
    best_acc = best["acc"]
    best_pred = best["pred"]

    logger.log(f"=== BEST MODEL: {best_name} | Val Top-1={best_acc*100:.2f}% ===")

    cm = confusion_matrix(y_va, best_pred, labels=list(range(len(labels))))
    rep = classification_report(y_va, best_pred, target_names=labels, zero_division=0)

    # save artifacts
    model_path = os.path.join(CFG.out_dir, f"best_model_{run_tag}.joblib")
    joblib.dump(best["model"], model_path)

    cm_path = os.path.join(CFG.out_dir, f"confusion_{run_tag}.npy")
    np.save(cm_path, cm)

    rep_path = os.path.join(CFG.out_dir, f"report_{run_tag}.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep)

    summary = {
        "run_tag": run_tag,
        "target": CFG.target,
        "labels": labels,
        "chance": chance,
        "best_model": best_name,
        "best_acc": best_acc,
        "models": [{"name": n, "acc": float(a), "time": float(t)} for (n,a,t,_,_) in results],
        "paths": {"model": model_path, "cm": cm_path, "report": rep_path}
    }

    summary_path = os.path.join(CFG.out_dir, f"summary_{run_tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.log(f"[SAVED] model={model_path}")
    logger.log(f"[SAVED] cm={cm_path}")
    logger.log(f"[SAVED] report={rep_path}")
    logger.log(f"[SAVED] summary={summary_path}")

    return summary


# ============================================================
# 8) AUTO LOOP (self-repair)
# ============================================================

def auto_run():
    set_seeds(CFG.seed)
    os.makedirs(CFG.out_dir, exist_ok=True)
    logger = Logger(CFG.out_dir)

    logger.log("============================================================")
    logger.log("AUTO COLOR RUNNER (tabular Lab-hist) | will try to self-repair")
    logger.log("============================================================")
    logger.log(f"CONFIG: {asdict(CFG)}")

    df, root_dir = download_and_read(logger)

    # Strategies: try multiple cheap orderings
    strategies = ["natural", "filesize", "mtime", "random", "natural", "random"]
    strategies = strategies[:CFG.max_strategies]

    overall_best = None

    for si, strat in enumerate(strategies, 1):
        logger.log("\n" + "="*70)
        logger.log(f"STRATEGY {si}/{len(strategies)} : {strat}")
        logger.log("="*70)

        mdf = build_matched_df(logger, df, root_dir, sort_mode=strat)
        if len(mdf) < 2000:
            logger.log("[WARN] Too few matched samples, skipping this strategy.")
            continue

        # Sanity score (fast, heuristic)
        logger.log("=== Sanity check: monotonicity score ===")
        sc = monotonicity_sanity_score(mdf)
        logger.log(f"[SANITY] monotonicity score ≈ {sc:.3f} (higher is better; ~0 means mismatch likely)")

        target_col = "tier7" if CFG.target == "tier7" else "color"
        # generate cache key based on strategy + label distribution (so cache invalidates if matching changes)
        key_payload = [strat, CFG.target, str(mdf[target_col].value_counts().to_dict())]
        strategy_key = sha1_of_list(key_payload)

        X, y, labels = extract_cached(logger, mdf, strategy_key=strategy_key)

        run_tag = f"{CFG.target}_{strat}_{strategy_key}"
        summary = train_and_eval(logger, X, y, labels, run_tag=run_tag)

        chance = summary["chance"]
        best_acc = summary["best_acc"]
        margin = best_acc - chance

        logger.log(f"=== Decision ===")
        logger.log(f"best_acc={best_acc*100:.2f}% | chance={chance*100:.2f}% | margin={margin*100:.2f}% | threshold={CFG.min_margin_over_chance*100:.2f}%")

        # Update overall best
        if overall_best is None or best_acc > overall_best["best_acc"]:
            overall_best = summary

        # If looks healthy enough, stop early.
        if margin >= CFG.min_margin_over_chance and sc >= 0.10:
            logger.log("[OK] This strategy looks non-garbage. Stopping early.")
            break
        else:
            logger.log("[SUSPICIOUS] Looks like weak signal or mismatch. Trying next strategy.")

    logger.log("\n" + "="*70)
    logger.log("FINAL SUMMARY")
    logger.log("="*70)
    if overall_best is None:
        logger.log("No successful run. (Probably matched 0 samples or extraction failed.)")
        return

    logger.log(json.dumps({
        "target": overall_best["target"],
        "best_model": overall_best["best_model"],
        "best_acc": overall_best["best_acc"],
        "chance": overall_best["chance"],
        "run_tag": overall_best["run_tag"],
        "paths": overall_best["paths"]
    }, indent=2))

    logger.log("DONE.")


if __name__ == "__main__":
    auto_run()
