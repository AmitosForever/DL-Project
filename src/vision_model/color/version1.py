import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import kagglehub

# -----------------------------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------------------------
CONFIG = {
    'img_size': 456,
    'batch_size': 32,
    'epochs': 10,
    'lr': 1e-4,
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dataset_handle': 'aayushpurswani/diamond-images-dataset',
    
    # Target Definitions
    'cat_tasks': ['cut', 'color', 'clarity', 'polish', 'symmetry', 'fluorescence', 'shape'],
    'num_tasks': ['carat'],
    
    # Loss Weights
    'loss_weights': {
        'cut': 1.0, 'color': 1.0, 'clarity': 1.0, 
        'polish': 1.0, 'symmetry': 1.0, 'fluorescence': 1.0, 
        'shape': 1.0, 'carat': 5.0 
    }
}

def get_dataset_setup():
    """
    Downloads the dataset and locates the 'web_scraped' folder 
    which contains both the CSV and the shape subfolders.
    """
    print(f"Checking/Downloading dataset: {CONFIG['dataset_handle']}...")
    try:
        base_path = kagglehub.dataset_download(CONFIG['dataset_handle'])
        print(f"Base download path: {base_path}")
    except Exception as e:
        print("Error downloading. Make sure you have 'kagglehub' installed.")
        raise e

    # Find diamond_data.csv to locate the actual data root
    # Based on your image, it is inside "web_scraped"
    csv_files = glob.glob(os.path.join(base_path, "**", "diamond_data.csv"), recursive=True)
    
    if not csv_files:
        raise FileNotFoundError("Could not find 'diamond_data.csv' in the downloaded files.")
    
    csv_path = csv_files[0]
    data_root = os.path.dirname(csv_path) # This should be the 'web_scraped' folder
    
    print(f"Data Root found at: {data_root}")
    return csv_path, data_root

# -----------------------------------------------------------------------------
# 2. Robust Dataset Class (Shape-based Matching)
# -----------------------------------------------------------------------------
class DiamondDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, encoders=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # --- 1. Load and Clean CSV ---
        try:
            self.df = pd.read_csv(csv_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")

        # Clean column names
        self.df.columns = [c.lower().strip() for c in self.df.columns]
        
        # Fix 'colour' -> 'color'
        if 'colour' in self.df.columns:
            print("Renaming column 'colour' to 'color'...")
            self.df.rename(columns={'colour': 'color'}, inplace=True)
            
        # Verify columns
        required_cols = CONFIG['cat_tasks'] + CONFIG['num_tasks']
        # We need 'shape' for matching logic, ensure it's checked
        if 'shape' not in required_cols: required_cols.append('shape')
             
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
            
        # Remove incomplete rows
        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)

        # --- 2. Match CSV Rows to Image Files ---
        # Logic: 
        # 1. The CSV has rows for 'Round', 'Emerald', etc.
        # 2. The folders are named 'round', 'emerald', etc.
        # 3. We pair the i-th row of 'Round' in CSV with the i-th file in 'round' folder.
        
        print("Matching images to metadata...")
        matched_data = []
        
        # Get all shape folders in the root directory
        # e.g. {'round': '/path/to/round', 'oval': '/path/to/oval'}
        available_folders = {}
        for d in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, d)
            if os.path.isdir(full_path):
                available_folders[d.lower()] = full_path

        # Helper for natural sorting (diamond_1.jpg, diamond_2.jpg, ... diamond_10.jpg)
        def natural_key(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

        unique_shapes = self.df['shape'].unique()
        
        for shape_name in unique_shapes:
            shape_str = str(shape_name).lower()
            
            # Check if we have a folder for this shape
            if shape_str not in available_folders:
                print(f"Warning: No folder found for shape '{shape_name}'. Skipping these rows.")
                continue
                
            folder_path = available_folders[shape_str]
            
            # List all images in that folder
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            # SORTING IS CRITICAL for alignment
            images.sort(key=natural_key)
            
            # Get corresponding CSV rows
            shape_df_indices = self.df.index[self.df['shape'].astype(str).str.lower() == shape_str].tolist()
            
            # We can only match up to the smaller count (files vs rows)
            limit = min(len(images), len(shape_df_indices))
            
            if limit < len(shape_df_indices):
                print(f"Warning: Shape '{shape_name}' has {len(shape_df_indices)} rows but only {len(images)} images. Truncating.")
            
            for i in range(limit):
                original_idx = shape_df_indices[i]
                img_path = os.path.join(folder_path, images[i])
                
                # Create a record combining the row data and the image path
                record = self.df.iloc[original_idx].to_dict()
                record['image_path'] = img_path
                matched_data.append(record)

        if not matched_data:
            raise RuntimeError("No valid image-label pairs found! Check if folder names match 'shape' column values.")

        # Create the final clean DataFrame
        self.final_df = pd.DataFrame(matched_data)
        print(f"Successfully matched {len(self.final_df)} images out of {len(self.df)} original rows.")

        # --- 3. Encoders ---
        self.cat_tasks = CONFIG['cat_tasks']
        self.num_tasks = CONFIG['num_tasks']
        self.encoders = {}

        if encoders is None:
            for task in self.cat_tasks:
                le = LabelEncoder()
                # Convert to string to be safe
                self.final_df[task] = self.final_df[task].astype(str)
                self.final_df[task] = le.fit_transform(self.final_df[task])
                self.encoders[task] = le
        else:
            self.encoders = encoders
            for task in self.cat_tasks:
                self.final_df[task] = self.final_df[task].astype(str)
                known = set(self.encoders[task].classes_)
                # Handle unknown classes by mapping to the first known class
                self.final_df[task] = self.final_df[task].apply(lambda x: x if x in known else list(known)[0])
                self.final_df[task] = self.encoders[task].transform(self.final_df[task])

    def __len__(self):
        return len(self.final_df)

    def __getitem__(self, idx):
        row = self.final_df.iloc[idx]
        img_path = row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception:
            # Return black image on corruption to prevent crash
            image = torch.zeros((3, CONFIG['img_size'], CONFIG['img_size']))

        targets = {}
        for task in self.cat_tasks:
            targets[task] = torch.tensor(row[task], dtype=torch.long)
        for task in self.num_tasks:
            targets[task] = torch.tensor(row[task], dtype=torch.float32)

        return image, targets

    def get_num_classes(self):
        return {task: len(le.classes_) for task, le in self.encoders.items()}

# -----------------------------------------------------------------------------
# 3. Multi-Task Model
# -----------------------------------------------------------------------------
class MultiTaskDiamondResNet(nn.Module):
    def __init__(self, task_classes):
        super(MultiTaskDiamondResNet, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.backbone = models.resnet50(weights=weights)
        
        # Remove original FC
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.heads = nn.ModuleDict()
        
        # Classification Heads
        for task, num_classes in task_classes.items():
            self.heads[task] = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        # Regression Head (Carat)
        self.heads['carat'] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        outputs = {}
        for task in CONFIG['cat_tasks']:
            outputs[task] = self.heads[task](features)
        
        outputs['carat'] = self.heads['carat'](features).squeeze(-1)
        return outputs

# -----------------------------------------------------------------------------
# 4. Training Engine
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda')
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()

    def calculate_loss(self, outputs, targets):
        total_loss = 0
        losses = {}
        
        # Categorical Loss
        for task in CONFIG['cat_tasks']:
            l = self.criterion_cls(outputs[task], targets[task].to(self.device))
            total_loss += l * CONFIG['loss_weights'][task]
            losses[task] = l.item()

        # Numerical Loss
        for task in CONFIG['num_tasks']:
            l = self.criterion_reg(outputs[task], targets[task].to(self.device))
            total_loss += l * CONFIG['loss_weights'][task]
            losses[task] = l.item()

        return total_loss, losses

    def train_epoch(self):
        self.model.train()
        running_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, targets in pbar:
            images = images.to(self.device)
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss, _ = self.calculate_loss(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return running_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        metrics = {t: 0 for t in CONFIG['cat_tasks']}
        metrics['carat_mae'] = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                
                bs = images.size(0)
                total += bs
                
                # Accuracy
                for task in CONFIG['cat_tasks']:
                    preds = torch.argmax(outputs[task], dim=1)
                    target = targets[task].to(self.device)
                    metrics[task] += (preds == target).sum().item()
                
                # Regression MAE
                metrics['carat_mae'] += torch.abs(outputs['carat'] - targets['carat'].to(self.device)).sum().item()

        return {k: v / total for k, v in metrics.items()}

# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------
def main():
    # 1. Get Data
    csv_path, data_root = get_dataset_setup()
    
    # 2. Transforms
    train_tfm = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 3. Create Dataset
    full_ds = DiamondDataset(csv_path, data_root, transform=train_tfm)
    
    # 4. Split
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    # 5. Model Setup
    task_classes = full_ds.get_num_classes()
    print(f"Tasks Configured: {task_classes}")
    
    model = MultiTaskDiamondResNet(task_classes).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, CONFIG['device'])
    
    # 6. Run Training
    print(f"Starting training on {CONFIG['device']}...")
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        train_loss = trainer.train_epoch()
        val_metrics = trainer.evaluate()
        
        print(f"Train Loss: {train_loss:.4f}")
        print("Val Metrics:")
        for k, v in val_metrics.items():
            if 'mae' in k:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k} Acc: {v:.2%}")
                
        torch.save(model.state_dict(), 'diamond_multitask_model.pth')

if __name__ == '__main__':
    main()