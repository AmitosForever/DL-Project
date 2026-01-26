import os
import glob
import re
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
import kagglehub

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# 1. Configuration (STABILIZED)
# -----------------------------------------------------------------------------
CONFIG = {
    'img_size': 456,
    'batch_size': 32,
    'epochs': 15,          
    'lr_backbone': 1e-5,   
    'lr_heads': 1e-4,     
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dataset_handle': 'aayushpurswani/diamond-images-dataset',
    
    # Feature Engineering Params
    'roi_crop_scale': 0.60, 
    'lab_boost_factor': 1.5, 
    'label_smoothing': 0.0,  # DISABLED for stability
    
    'cat_tasks': ['cut', 'clarity', 'polish', 'symmetry', 'fluorescence', 'shape'], 
    'ordinal_tasks': ['color'], 
    'num_tasks': ['carat'],
    
    # REDUCED Loss Weights to keep total loss O(1-10)
    'loss_weights': {
        'cut': 1.0, 
        'color': 0.5,      # Reduced from 2.0 (Regression loss scale is different)
        'clarity': 1.0,    # Reduced from 2.0
        'polish': 1.0, 
        'symmetry': 1.0, 
        'fluorescence': 1.0, 
        'shape': 1.0,      
        'carat': 1.0       # Reduced from 5.0 (MSE can be large, we don't want it to dominate)
    },
    
    'unfreeze_epoch': 3    # Extended freeze duration (Epoch 0, 1, 2 frozen)
}

# -----------------------------------------------------------------------------
# 2. Advanced Transforms
# -----------------------------------------------------------------------------
class LabColorBoost(object):
    def __init__(self, boost_factor=1.5):
        self.boost = boost_factor

    def __call__(self, img):
        return transforms.functional.adjust_saturation(img, self.boost)

class CenterROICrop(object):
    def __init__(self, scale=0.7):
        self.scale = scale

    def __call__(self, img):
        w, h = img.size
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        return img.crop((left, top, left + new_w, top + new_h))

def get_transforms(split='train'):
    t_list = []
    t_list.append(CenterROICrop(scale=CONFIG['roi_crop_scale']))
    if split == 'train':
        t_list.append(LabColorBoost(boost_factor=CONFIG['lab_boost_factor']))
        t_list.append(transforms.RandomHorizontalFlip())
    t_list.append(transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])))
    t_list.append(transforms.ToTensor())
    t_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(t_list)

# -----------------------------------------------------------------------------
# 3. Robust Dataset & Normalized Class Weights
# -----------------------------------------------------------------------------
class DiamondDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, encoders=None):
        self.root_dir = root_dir
        self.transform = transform
        
        try:
            self.df = pd.read_csv(csv_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")
        
        self.df.columns = [c.lower().strip() for c in self.df.columns]
        if 'colour' in self.df.columns:
            self.df.rename(columns={'colour': 'color'}, inplace=True)
            
        required_cols = CONFIG['cat_tasks'] + CONFIG['ordinal_tasks'] + CONFIG['num_tasks'] + ['shape']
        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)

        print("Mapping images...")
        available_folders = {d.lower(): os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))}
        
        def natural_key(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

        matched_data = []
        for shape_name in self.df['shape'].unique():
            shape_str = str(shape_name).lower()
            if shape_str not in available_folders: continue
            
            folder_path = available_folders[shape_str]
            images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.png'))], key=natural_key)
            
            shape_df_indices = self.df.index[self.df['shape'].str.lower() == shape_str].tolist()
            limit = min(len(images), len(shape_df_indices))
            
            for i in range(limit):
                rec = self.df.iloc[shape_df_indices[i]].to_dict()
                rec['image_path'] = os.path.join(folder_path, images[i])
                matched_data.append(rec)
                
        self.df = pd.DataFrame(matched_data)
        print(f"Dataset Size: {len(self.df)}")

        self.encoders = encoders if encoders else {}
        self.class_weights = {} 

        # 1. Standard Categorical
        for task in CONFIG['cat_tasks']:
            self.df[task] = self.df[task].astype(str)
            if task not in self.encoders:
                le = LabelEncoder()
                self.df[task] = le.fit_transform(self.df[task])
                self.encoders[task] = le
            else:
                known = set(self.encoders[task].classes_)
                self.df[task] = self.df[task].apply(lambda x: x if x in known else list(known)[0])
                self.df[task] = self.encoders[task].transform(self.df[task])
            
            # --- WEIGHT NORMALIZATION FIX ---
            if encoders is None: 
                counts = Counter(self.df[task])
                total = sum(counts.values())
                num_classes = len(self.encoders[task].classes_)
                
                # Raw Inverse Frequency
                raw_weights = torch.zeros(num_classes)
                for cls_idx, count in counts.items():
                    raw_weights[cls_idx] = total / (num_classes * count)
                
                # Normalize: Mean should be 1.0
                mean_weight = raw_weights.mean()
                normalized_weights = raw_weights / mean_weight
                
                # Clamp: Prevent extreme multipliers (e.g., 50x for rare classes)
                clamped_weights = torch.clamp(normalized_weights, min=0.5, max=5.0)
                
                self.class_weights[task] = clamped_weights

        # 2. Ordinal Color 
        color_order = sorted(self.df['color'].unique()) 
        self.color_map = {c: i for i, c in enumerate(color_order)}
        self.df['color_ordinal'] = self.df['color'].map(self.color_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except:
            img = torch.zeros((3, CONFIG['img_size'], CONFIG['img_size']))

        targets = {}
        for task in CONFIG['cat_tasks']:
            targets[task] = torch.tensor(row[task], dtype=torch.long)
        
        targets['color'] = torch.tensor(row['color_ordinal'], dtype=torch.float32)
        targets['carat'] = torch.tensor(row['carat'], dtype=torch.float32)

        return img, targets

    def get_num_classes(self):
        return {task: len(le.classes_) for task, le in self.encoders.items()}

# -----------------------------------------------------------------------------
# 4. Multi-Task Model
# -----------------------------------------------------------------------------
class EnhancedDiamondNet(nn.Module):
    def __init__(self, task_classes):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.backbone = models.resnet50(weights=weights)
        
        # Initial freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.heads = nn.ModuleDict()
        
        for task in CONFIG['cat_tasks']:
            self.heads[task] = nn.Sequential(
                nn.Linear(num_fts, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, task_classes[task])
            )
            
        self.heads['color'] = nn.Sequential(
            nn.Linear(num_fts, 512),
            nn.ReLU(),
            nn.Linear(512, 1) 
        )
        
        self.heads['carat'] = nn.Sequential(
            nn.Linear(num_fts, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def unfreeze_backbone(self):
        print(">>> Unfreezing Backbone Layers 3 & 4...")
        for name, child in self.backbone.named_children():
            if name in ['layer3', 'layer4']:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x):
        feat = self.backbone(x)
        out = {}
        for task in CONFIG['cat_tasks']:
            out[task] = self.heads[task](feat)
        
        out['color'] = self.heads['color'](feat).squeeze(-1)
        out['carat'] = self.heads['carat'](feat).squeeze(-1)
        return out

# -----------------------------------------------------------------------------
# 5. Training Engine
# -----------------------------------------------------------------------------
def train_model():
    path = kagglehub.dataset_download(CONFIG['dataset_handle'])
    csv_files = glob.glob(os.path.join(path, "**", "diamond_data.csv"), recursive=True)
    csv_path = csv_files[0]
    root_dir = os.path.dirname(csv_path)

    full_ds = DiamondDataset(csv_path, root_dir, transform=get_transforms('train'))
    
    train_len = int(0.8 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=CONFIG['num_workers'], pin_memory=True)

    task_classes = full_ds.get_num_classes()
    model = EnhancedDiamondNet(task_classes).to(CONFIG['device'])
    
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CONFIG['lr_backbone']},
        {'params': model.heads.parameters(), 'lr': CONFIG['lr_heads']}
    ])
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Losses
    criteria = {}
    for task in CONFIG['cat_tasks']:
        w = full_ds.class_weights[task].to(CONFIG['device'])
        # LABEL SMOOTHING DISABLED
        criteria[task] = nn.CrossEntropyLoss(weight=w, label_smoothing=CONFIG['label_smoothing'])
    
    crit_reg = nn.MSELoss() 

    print("Starting Training (Stabilized)...")
    
    for epoch in range(CONFIG['epochs']):
        # Unfreeze Schedule (Extended)
        if epoch == CONFIG['unfreeze_epoch']:
            model.unfreeze_backbone()
        
        # Shape Decay
        shape_w = CONFIG['loss_weights']['shape'] * (0.5 ** epoch)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Shape W: {shape_w:.4f}")
        
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for imgs, targets in pbar:
            imgs = imgs.to(CONFIG['device'])
            targets = {k: v.to(CONFIG['device']) for k, v in targets.items()}
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                total_loss = 0
                
                for task in CONFIG['cat_tasks']:
                    w = shape_w if task == 'shape' else CONFIG['loss_weights'][task]
                    l = criteria[task](preds[task], targets[task])
                    total_loss += l * w
                
                l_col = crit_reg(preds['color'], targets['color'])
                total_loss += l_col * CONFIG['loss_weights']['color']
                
                l_car = crit_reg(preds['carat'], targets['carat'])
                total_loss += l_car * CONFIG['loss_weights']['carat']
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})
            
        # Validation
        model.eval()
        metrics = {k: 0 for k in CONFIG['cat_tasks']}
        metrics.update({'color_top1': 0, 'color_top3': 0, 'carat_mae': 0})
        total = 0
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(CONFIG['device'])
                targets = {k: v.to(CONFIG['device']) for k, v in targets.items()}
                
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                
                bs = imgs.size(0)
                total += bs
                
                for task in CONFIG['cat_tasks']:
                    p_cls = torch.argmax(preds[task], dim=1)
                    metrics[task] += (p_cls == targets[task]).sum().item()
                
                col_pred_rank = torch.round(preds['color'])
                col_diff = torch.abs(col_pred_rank - targets['color'])
                metrics['color_top1'] += (col_diff == 0).sum().item() 
                metrics['color_top3'] += (col_diff <= 1).sum().item() 
                
                metrics['carat_mae'] += torch.abs(preds['carat'] - targets['carat']).sum().item()
        
        print(f"--- Val Results Ep {epoch+1} ---")
        for k, v in metrics.items():
            val = v / total
            if 'mae' in k:
                print(f"{k}: {val:.4f}")
            else:
                print(f"{k}: {val:.2%}")
        
        torch.save(model.state_dict(), f"diamond_net_ep{epoch+1}.pth")

if __name__ == '__main__':
    train_model()