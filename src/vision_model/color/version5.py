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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
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
    sog_p: int = 6  # Shades-of-Gray p-norm

    # Task
    target: str = "tier7"  # "tier7" or "raw17"
    val_size: float = 0.20

    # Auto logic
    max_strategies: int = 4
    # Decision thresholds to detect "fake success"
    min_gain_over_majority: float = 0.02  # +2% over always-majority
    min_balanced_acc: float = 0.22        # for 7 classes, chance bal-acc ~ 0.142
    cache_dir: str = "./_auto_cache"
    out_dir: str = "./_auto_out"

    # Models
    lr_max_iter: int = 6000
    hgb_max_iter: int = 600
    rf_n_estimators: int = 500

CFG = Config()


# ============================================================
# 1) LOGGING
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

def sha1_of_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


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

    df["color"] = df["color"].apply(clean_color_label)
    df = df[df["color"].isin(VALID_GRADES_17)].copy()

    logger.log("=== Raw color distribution ===")
    logger.log(str(df["color"].value_counts()))

    df["tier7"] = df["color"].apply(map_color_to_tier7)
    logger.log("=== Tier7 distribution (CSV) ===")
    logger.log(str(df["tier7"].value_counts()))

    return df, root_dir


# ============================================================
# 5) MATCH STRATEGIES
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
        p2 = paths[:]
        rng.shuffle(p2)
        return p2
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

        for i in range(n):
            rec = rows.iloc[i].to_dict()
            rec["image_path"] = imgs[i]
            matched.append(rec)

    mdf = pd.DataFrame(matched)
    logger.log(f"[DATA] Matched samples: {len(mdf)}")
    if len(mdf) == 0:
        return mdf

    target_col = "tier7" if CFG.target == "tier7" else "color"
    logger.log(f"=== Distribution after matching ({target_col}) ===")
    logger.log(str(mdf[target_col].value_counts()))
    return mdf


# ============================================================
# 6) FEATURE EXTRACTION (per-image cache)
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

    # ignore extreme glare/shadow
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

    ratio = (bm - 128.0) / (Lm + 1e-6)
    ratio_stats = np.array([
        ratio.mean(), ratio.std(),
        np.percentile(ratio, 25),
        np.percentile(ratio, 50),
        np.percentile(ratio, 75)
    ], dtype=np.float32)

    return np.concatenate([hist_b, hist_a, hist_L, stats, ratio_stats], axis=0)

def per_image_cache_paths() -> Tuple[str, str]:
    os.makedirs(CFG.cache_dir, exist_ok=True)
    key = f"imgfeat_sz{CFG.img_size}_b{CFG.bins_b}_a{CFG.bins_a}_l{CFG.bins_l}_p{CFG.sog_p}"
    h = sha1_of_str(key)
    npz_path = os.path.join(CFG.cache_dir, f"image_feature_cache_{h}.npz")
    meta_path = os.path.join(CFG.cache_dir, f"image_feature_cache_{h}_meta.json")
    return npz_path, meta_path

def load_or_build_image_feature_cache(logger: Logger, image_paths: List[str]) -> Tuple[np.ndarray, List[str], Dict[str,int]]:
    """
    Cache stores features for each image_path.
    Returns:
      X_all: (N, D)
      paths_all: list length N
      path_to_idx: mapping for fast lookup
    """
    npz_path, meta_path = per_image_cache_paths()
    D_expected = CFG.bins_b + CFG.bins_a + CFG.bins_l + 6 + 5

    # load existing
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        X_all = data["X"]
        paths_all = data["paths"].tolist()
        path_to_idx = {p:i for i,p in enumerate(paths_all)}
        logger.log(f"[IMG-CACHE] Loaded {len(paths_all)} images from {npz_path} | X={X_all.shape}")

        # compute missing
        missing = [p for p in image_paths if p not in path_to_idx]
        if not missing:
            return X_all, paths_all, path_to_idx

        logger.log(f"[IMG-CACHE] Missing {len(missing)} images. Extracting only missing...")
        new_feats = []
        new_paths = []
        bad = 0
        t0 = time.time()
        for p in tqdm(missing, total=len(missing)):
            f = extract_features(p)
            if f is None or f.shape[0] != D_expected:
                bad += 1
                continue
            new_feats.append(f)
            new_paths.append(p)
        logger.log(f"[IMG-CACHE] Missing extraction done in {time.time()-t0:.1f}s | bad={bad}")

        if new_feats:
            X_new = np.vstack(new_feats).astype(np.float32)
            X_all2 = np.vstack([X_all, X_new])
            paths_all2 = paths_all + new_paths
        else:
            X_all2 = X_all
            paths_all2 = paths_all

        np.savez_compressed(npz_path, X=X_all2, paths=np.array(paths_all2, dtype=object))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"count": len(paths_all2), "dim": int(X_all2.shape[1])}, f, indent=2)

        logger.log(f"[IMG-CACHE] Updated -> {npz_path} | total={len(paths_all2)}")
        path_to_idx2 = {p:i for i,p in enumerate(paths_all2)}
        return X_all2, paths_all2, path_to_idx2

    # build from scratch
    logger.log("[IMG-CACHE] Building per-image feature cache from scratch...")
    t0 = time.time()
    feats = []
    paths_ok = []
    bad = 0
    for p in tqdm(image_paths, total=len(image_paths)):
        f = extract_features(p)
        if f is None or f.shape[0] != D_expected:
            bad += 1
            continue
        feats.append(f)
        paths_ok.append(p)
    X_all = np.vstack(feats).astype(np.float32)
    np.savez_compressed(npz_path, X=X_all, paths=np.array(paths_ok, dtype=object))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(paths_ok), "dim": int(X_all.shape[1]), "bad": bad}, f, indent=2)
    logger.log(f"[IMG-CACHE] Built in {time.time()-t0:.1f}s | ok={len(paths_ok)} bad={bad} | X={X_all.shape}")
    path_to_idx = {p:i for i,p in enumerate(paths_ok)}
    return X_all, paths_ok, path_to_idx


def build_Xy_from_matched(logger: Logger, mdf: pd.DataFrame, X_all: np.ndarray, path_to_idx: Dict[str,int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    target_col = "tier7" if CFG.target == "tier7" else "color"
    labels = sorted(mdf[target_col].unique().tolist())
    lab2idx = {c:i for i,c in enumerate(labels)}

    X_list = []
    y_list = []
    missing = 0
    for _, row in mdf.iterrows():
        p = row["image_path"]
        if p not in path_to_idx:
            missing += 1
            continue
        X_list.append(X_all[path_to_idx[p]])
        y_list.append(lab2idx[row[target_col]])

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    if missing:
        logger.log(f"[WARN] {missing} images missing from cache (bad images etc). Dropped.")
    logger.log(f"[DATASET] X={X.shape} y={y.shape} classes={len(labels)} labels={labels}")

    return X, y, labels


# ============================================================
# 7) TRAIN + EVAL (with weights + real metrics)
# ============================================================

def inv_freq_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w_class = 1.0 / counts
    w_class *= (num_classes / w_class.sum())  # normalize-ish
    return w_class[y]

def majority_baseline_acc(y: np.ndarray) -> float:
    counts = np.bincount(y)
    return float(counts.max() / counts.sum())

def train_and_eval(logger: Logger, X: np.ndarray, y: np.ndarray, labels: List[str], run_tag: str) -> Dict:
    os.makedirs(CFG.out_dir, exist_ok=True)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=CFG.val_size,
        random_state=CFG.seed,
        stratify=y
    )

    ncls = len(labels)
    chance = 1.0 / ncls
    maj = majority_baseline_acc(y_va)

    logger.log("=== Split + baselines ===")
    logger.log(f"[SPLIT] Train={len(y_tr)} | Val={len(y_va)} | Classes={ncls}")
    logger.log(f"[BASE] Chance Top-1 ≈ {chance*100:.2f}%")
    logger.log(f"[BASE] Majority Top-1 ≈ {maj*100:.2f}%  (predict always '{labels[int(np.bincount(y_va).argmax())]}')")

    sw_tr = inv_freq_weights(y_tr, ncls)

    results = []
    best = {"score": -1.0}

    def eval_preds(name: str, pred: np.ndarray, dur: float, model_obj):
        acc = accuracy_score(y_va, pred)
        bacc = balanced_accuracy_score(y_va, pred)
        mf1 = f1_score(y_va, pred, average="macro", zero_division=0)
        logger.log(f"[{name}] Acc={acc*100:.2f}% | BAcc={bacc*100:.2f}% | MacroF1={mf1:.3f} | time={dur:.1f}s")
        results.append((name, acc, bacc, mf1, dur, model_obj, pred))

    # Model 1: LR (needs weights)
    logger.log("=== MODEL 1: Scaled LogisticRegression (weighted) ===")
    t0 = time.time()
    model_lr = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=CFG.lr_max_iter,
            solver="lbfgs",
            n_jobs=-1,
            multi_class="auto"
        ))
    ])
    model_lr.fit(X_tr, y_tr, clf__sample_weight=sw_tr)
    pred_lr = model_lr.predict(X_va)
    eval_preds("LR", pred_lr, time.time()-t0, model_lr)

    # Model 2: HGB (supports sample_weight)
    logger.log("=== MODEL 2: HistGradientBoosting (weighted) ===")
    t0 = time.time()
    model_hgb = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_iter=CFG.hgb_max_iter,
        random_state=CFG.seed
    )
    model_hgb.fit(X_tr, y_tr, sample_weight=sw_tr)
    pred_hgb = model_hgb.predict(X_va)
    eval_preds("HGB", pred_hgb, time.time()-t0, model_hgb)

    # Model 3: RF (weighted)
    logger.log("=== MODEL 3: RandomForest (weighted) ===")
    t0 = time.time()
    model_rf = RandomForestClassifier(
        n_estimators=CFG.rf_n_estimators,
        random_state=CFG.seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model_rf.fit(X_tr, y_tr, sample_weight=sw_tr)
    pred_rf = model_rf.predict(X_va)
    eval_preds("RF", pred_rf, time.time()-t0, model_rf)

    # choose best by Balanced Accuracy first, then MacroF1, then Acc
    results_sorted = sorted(results, key=lambda t: (t[2], t[3], t[1]), reverse=True)
    best_name, best_acc, best_bacc, best_mf1, best_time, best_model, best_pred = results_sorted[0]

    logger.log(f"=== BEST (by BAcc/MacroF1): {best_name} ===")

    cm = confusion_matrix(y_va, best_pred, labels=list(range(ncls)))
    rep = classification_report(y_va, best_pred, target_names=labels, zero_division=0)

    model_path = os.path.join(CFG.out_dir, f"best_model_{run_tag}.joblib")
    joblib.dump(best_model, model_path)
    cm_path = os.path.join(CFG.out_dir, f"confusion_{run_tag}.npy")
    np.save(cm_path, cm)
    rep_path = os.path.join(CFG.out_dir, f"report_{run_tag}.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep)

    summary = {
        "run_tag": run_tag,
        "target": CFG.target,
        "labels": labels,
        "chance_acc": chance,
        "majority_acc": maj,
        "best_model": best_name,
        "best_acc": float(best_acc),
        "best_bacc": float(best_bacc),
        "best_macrof1": float(best_mf1),
        "models": [
            {"name": n, "acc": float(a), "bacc": float(b), "macrof1": float(m), "time": float(t)}
            for (n,a,b,m,t,_,_) in results
        ],
        "paths": {"model": model_path, "cm": cm_path, "report": rep_path}
    }
    summary_path = os.path.join(CFG.out_dir, f"summary_{run_tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.log(f"[SAVED] model={model_path}")
    logger.log(f"[SAVED] report={rep_path}")
    logger.log(f"[SAVED] summary={summary_path}")

    # "Fake success" detector
    gain_over_maj = best_acc - maj
    logger.log(f"=== Reality check ===")
    logger.log(f"Gain over majority = {gain_over_maj*100:.2f}% | BAcc={best_bacc*100:.2f}% | thresholds: +{CFG.min_gain_over_majority*100:.2f}%, BAcc>={CFG.min_balanced_acc*100:.2f}%")

    if gain_over_maj < CFG.min_gain_over_majority or best_bacc < CFG.min_balanced_acc:
        logger.log("[DIAG] This is likely NOT learning meaningful color tiers (probably majority-class or label-image mismatch).")
        logger.log("[DIAG] If you need certainty, this dataset/matching method is not usable for 7-tier classification from images alone.")
    else:
        logger.log("[DIAG] There is at least some non-trivial signal beyond majority baseline.")

    return summary


# ============================================================
# 8) MAIN AUTO LOOP
# ============================================================

def auto_run():
    set_seeds(CFG.seed)
    logger = Logger(CFG.out_dir)

    logger.log("============================================================")
    logger.log("AUTO COLOR RUNNER v5 (real metrics + per-image cache)")
    logger.log("============================================================")
    logger.log(f"CONFIG: {asdict(CFG)}")

    df, root_dir = download_and_read(logger)

    strategies = ["natural", "filesize", "mtime", "random"][:CFG.max_strategies]

    overall_best = None

    for i, strat in enumerate(strategies, 1):
        logger.log("\n" + "="*70)
        logger.log(f"STRATEGY {i}/{len(strategies)} : {strat}")
        logger.log("="*70)

        mdf = build_matched_df(logger, df, root_dir, sort_mode=strat)
        if len(mdf) < 2000:
            logger.log("[WARN] Too few matched samples, skipping.")
            continue

        # Build / update per-image feature cache only for paths we need
        paths_needed = mdf["image_path"].astype(str).tolist()
        X_all, paths_all, path_to_idx = load_or_build_image_feature_cache(logger, paths_needed)

        X, y, labels = build_Xy_from_matched(logger, mdf, X_all, path_to_idx)

        run_tag = f"{CFG.target}_{strat}_{sha1_of_str(strat + '_' + str(len(X)))}"
        summary = train_and_eval(logger, X, y, labels, run_tag)

        # track overall best by balanced accuracy
        if overall_best is None or summary["best_bacc"] > overall_best["best_bacc"]:
            overall_best = summary

    logger.log("\n" + "="*70)
    logger.log("FINAL SUMMARY (best by balanced accuracy)")
    logger.log("="*70)
    if overall_best is None:
        logger.log("No successful run.")
        return

    logger.log(json.dumps({
        "best_model": overall_best["best_model"],
        "best_acc": overall_best["best_acc"],
        "best_bacc": overall_best["best_bacc"],
        "best_macrof1": overall_best["best_macrof1"],
        "majority_acc": overall_best["majority_acc"],
        "paths": overall_best["paths"]
    }, indent=2))
    logger.log("DONE.")

if __name__ == "__main__":
    auto_run()
