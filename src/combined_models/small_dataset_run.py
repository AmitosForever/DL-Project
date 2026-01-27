import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torchvision import models, transforms

# אפשר טעינת תמונות קטועות למניעת קריסות
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# הגדרות (Configuration)
# =========================
ROOT = Path(__file__).resolve().parent
EXCEL_IN = ROOT / "data" / "Diamonds" / "diamonds_clean_combined.xlsx"
EXCEL_OUT = ROOT / "data" / "final_check_with_preds.xlsx"
IMAGES_ROOT = ROOT / "data" / "Diamonds"
MODELS_DIR = ROOT / "vision_models"

# עמודות באקסל
ID_COL = "Id"
FOLDER_COL = "folder_name"

# =========================
# נתיבי המודלים
# =========================
MODELS_PATHS = {
    "Carat":    MODELS_DIR / "carat.pt",
    "Clarity":  MODELS_DIR / "clarity.pt",
    "Color":    MODELS_DIR / "color.pth",
    "Cut":      MODELS_DIR / "cut.pt",
    "Polish":   MODELS_DIR / "polish.pt",
    "Shape":    MODELS_DIR / "shape.pt",
    "Symmetry": MODELS_DIR / "symmetry.pt",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# =========================
# הגדרות ארכיטקטורה (Architecture Definitions)
# =========================

class CaratNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)

class ConvNeXtTinyHead(nn.Module):
    """עטיפה למודלים המשתמשים ב-ConvNeXt-Tiny (Clarity, Cut, Polish, Symmetry)"""
    def __init__(self, num_classes, is_binary=False):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = models.convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Dropout(p=0.0), 
            nn.Linear(in_features, num_classes),
        )
        self.is_binary = is_binary

    def forward(self, x):
        out = self.backbone(x)
        if self.is_binary:
            return out.squeeze(1)
        return out

class ResNet18Classifier(nn.Module):
    """עטיפה למודל Shape"""
    def __init__(self, num_classes):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def make_color_model(num_classes):
    """EfficientNet V2 S עבור Color"""
    base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_feats = base.classifier[1].in_features
    base.classifier = nn.Sequential(nn.Dropout(0.0), nn.Linear(in_feats, num_classes))
    return base

# =========================
# פונקציות עזר (Utils)
# =========================

def shades_of_gray_wb(img_np, p=6, eps=1e-6):
    """איזון לבן (White Balance) ייעודי למודל הצבע"""
    img = np.clip(img_np, 0.0, 1.0)
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p) + eps
    illum = illum / (np.mean(illum) + eps)
    out = img / illum[None, None, :]
    return np.clip(out, 0.0, 1.0)

def find_image(folder, img_id):
    """מוצא תמונה בתיקייה עם סיומות שונות"""
    base = IMAGES_ROOT / str(folder)
    if not base.exists():
        return None
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        p = base / f"{img_id}{ext}"
        if p.exists():
            return p
    return None

def _clean_state_dict(state_dict):
    """מנקה prefix של 'module.' כדי למנוע שגיאות טעינה"""
    new_sd = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k.startswith("backbone.model."): 
            k = k.replace("backbone.model.", "backbone.") 
        new_sd[k] = v
    return new_sd

def detect_output_size(state_dict, layer_name_suffix):
    """מזהה דינמית את גודל השכבה האחרונה מהמשקולות שנשמרו"""
    for k, v in state_dict.items():
        if k.endswith(layer_name_suffix):
            return v.shape[0]
    return None

# =========================
# טועני מודלים (Loaders)
# =========================

def load_carat_model(path):
    print(f"[LOAD] Carat: {path.name}")
    ckpt = torch.load(path, map_location=DEVICE)
    y_mean = ckpt.get("y_mean_t", 0.0)
    y_std = ckpt.get("y_std_t", 1.0)
    
    model = CaratNet()
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(_clean_state_dict(state), strict=False)
    model.to(DEVICE).eval()
    
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    def post_process(output_tensor):
        vals = output_tensor.view(-1).cpu().numpy()
        # Denormalize & Inverse Log1p
        vals = vals * y_std + y_mean
        return np.expm1(vals)

    return model, tfm, post_process

def load_polish_model(path):
    print(f"[LOAD] Polish: {path.name}")
    ckpt = torch.load(path, map_location=DEVICE)
    
    model = ConvNeXtTinyHead(num_classes=1, is_binary=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(_clean_state_dict(state), strict=False)
    model.to(DEVICE).eval()
    
    img_size = ckpt.get("train", {}).get("img_size_ft", 512)
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    threshold = ckpt.get("threshold", 0.5)
    
    def post_process(output_tensor):
        probs = torch.sigmoid(output_tensor).cpu().numpy()
        return ["EX" if p > threshold else "VG/GD" for p in probs]

    return model, tfm, post_process

def load_color_model(path):
    print(f"[LOAD] Color: {path.name}")
    state_dict = torch.load(path, map_location=DEVICE)
    state_dict = _clean_state_dict(state_dict)

    # 1. זיהוי מספר המחלקות בקובץ
    num_classes_in_file = detect_output_size(state_dict, "classifier.1.weight")
    if num_classes_in_file is None:
        num_classes_in_file = 7 
        print(f"  -> [WARN] Could not detect output size, defaulting to {num_classes_in_file}")
    else:
        print(f"  -> Detected {num_classes_in_file} classes in checkpoint.")

    # 2. הגדרת המחלקות בסדר אלפביתי (תיקון לבעיית ההיפוך)
    # PyTorch ImageFolder ממיין תיקיות לפי א-ב כברירת מחדל
    classes = sorted([
        "Premium_White",        # D-F
        "Near_Colorless_High",  # G-H
        "Near_Colorless_Low",   # I-J
        "Faint_Yellow",         # K-M
        "Very_Light_Yellow",    # N-R
        "Light_Yellow",         # S-Z
        "Yellow_LowEnd"         # Other
    ])
    
    # אם חסרה מחלקה אחת, נניח שזו האחרונה (הכי נדירה)
    if num_classes_in_file == 6:
        classes = [c for c in classes if c != "Yellow_LowEnd"]
        
    print(f"  -> Using Alphabetical Mapping: {classes}")

    # 3. אתחול המודל
    model = make_color_model(num_classes=len(classes))
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    
    def transform_fn(img_pil):
        img_pil = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
        x_np = np.array(img_pil).astype(np.float32) / 255.0
        x_wb = shades_of_gray_wb(x_np, p=6)
        x_t = torch.from_numpy(x_wb).permute(2, 0, 1)
        x_t = (x_t - 0.5) / 0.5
        return x_t

    def post_process(output_tensor):
        preds = output_tensor.argmax(1).cpu().numpy()
        return [classes[i] for i in preds]

    return model, transform_fn, post_process

def load_generic_classifier(task_name, path, architecture):
    print(f"[LOAD] {task_name}: {path.name}")
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt["model"] if "model" in ckpt else (ckpt["model_state"] if "model_state" in ckpt else ckpt)
    state = _clean_state_dict(state)

    # חילוץ שמות המחלקות מה-Metadata
    classes = None
    if isinstance(ckpt, dict):
        if "classes" in ckpt: classes = ckpt["classes"]
        elif "idx2label" in ckpt: 
            idx2 = ckpt["idx2label"]
            classes = [idx2[i] for i in range(len(idx2))]
        elif "sym_mapping" in ckpt:
            mapping = ckpt["sym_mapping"]
            classes = [None] * len(mapping)
            for k, v in mapping.items(): classes[v] = k
    
    if classes is None: raise ValueError(f"No classes found for {task_name}")

    # 1. זיהוי גודל הפלט האמיתי של המודל (לטיפול במקרי רגרסיה)
    layer_key = "backbone.classifier.3.weight" if architecture == "convnext" else "backbone.fc.weight"
    real_out_features = detect_output_size(state, layer_key)
    
    if real_out_features is None:
        real_out_features = len(classes)
    
    print(f"  -> Classes metadata: {len(classes)} labels.")
    print(f"  -> Model weights: {real_out_features} outputs.")

    # 2. אתחול המודל
    if architecture == "convnext":
        model = ConvNeXtTinyHead(num_classes=real_out_features)
    elif architecture == "resnet18":
        model = ResNet18Classifier(num_classes=real_out_features)
    
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    
    # 3. גודל תמונה
    img_size = 224
    if isinstance(ckpt, dict) and "train" in ckpt:
         img_size = ckpt["train"].get("img_size_ft", ckpt["train"].get("img_size", 224))
    elif "img_size" in ckpt: 
        img_size = ckpt["img_size"]
    print(f"  -> Using img_size: {img_size}")

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    def post_process(output_tensor):
        # טיפול ברגרסיה (Clarity) או סיווג
        if output_tensor.shape[1] == 1:
             vals = output_tensor.view(-1).cpu().numpy()
             idxs = np.clip(np.round(vals), 0, len(classes)-1).astype(int)
             return [classes[i] for i in idxs]
        else:
            preds = output_tensor.argmax(1).cpu().numpy()
            return [classes[i] for i in preds]

    return model, tfm, post_process

# =========================
# ריצה ראשית (Main)
# =========================

def main():
    print(f"[INFO] Running on device: {DEVICE}")
    print(f"[INFO] Models directory: {MODELS_DIR}")
    
    if not EXCEL_IN.exists():
        print(f"[ERROR] Excel not found: {EXCEL_IN}")
        return
        
    df = pd.read_excel(EXCEL_IN)
    print(f"[INFO] Loaded {len(df)} rows.")

    image_paths = []
    missing = 0
    for _, row in df.iterrows():
        p = find_image(row[FOLDER_COL], row[ID_COL])
        image_paths.append(p)
        if p is None: missing += 1
    print(f"[INFO] Missing images: {missing}")

    for task_name, model_path in MODELS_PATHS.items():
        if not model_path.exists():
            print(f"[SKIP] Missing: {model_path}")
            continue
            
        print(f"\n{'='*30}\nProcessing Task: {task_name}\n{'='*30}")
        
        try:
            # בחירת הלואדר המתאים
            if task_name == "Carat":
                model, tfm, post_proc = load_carat_model(model_path)
            elif task_name == "Polish":
                model, tfm, post_proc = load_polish_model(model_path)
            elif task_name == "Color":
                model, tfm, post_proc = load_color_model(model_path)
            elif task_name == "Shape":
                model, tfm, post_proc = load_generic_classifier(task_name, model_path, "resnet18")
            elif task_name in ["Clarity", "Cut", "Symmetry"]:
                model, tfm, post_proc = load_generic_classifier(task_name, model_path, "convnext")
            else:
                continue

            # לולאת הרצה
            preds = [np.nan] * len(df)
            buffer_imgs = []
            buffer_idxs = []
            
            for i, p in enumerate(tqdm(image_paths, desc=task_name)):
                if p is None: continue
                try:
                    img = Image.open(p).convert("RGB")
                    buffer_imgs.append(tfm(img))
                    buffer_idxs.append(i)
                except: continue
                
                if len(buffer_imgs) >= BATCH_SIZE:
                    with torch.no_grad():
                        out = model(torch.stack(buffer_imgs).to(DEVICE))
                        res = post_proc(out)
                        for idx, val in zip(buffer_idxs, res): preds[idx] = val
                    buffer_imgs, buffer_idxs = [], []
            
            if buffer_imgs:
                with torch.no_grad():
                    out = model(torch.stack(buffer_imgs).to(DEVICE))
                    res = post_proc(out)
                    for idx, val in zip(buffer_idxs, res): preds[idx] = val

            # שמירת התוצאות
            df[task_name] = preds
            df.to_excel(EXCEL_OUT, index=False)
            
        except Exception as e:
            print(f"[ERROR] Task {task_name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[DONE] Saved to: {EXCEL_OUT}")

if __name__ == "__main__":
    main()