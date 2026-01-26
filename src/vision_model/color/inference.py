# inference.py
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch import nn

# 1. Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "_final_model_out/best_model.pth" # Path to your saved file

def load_model():
    """Loads the model and the class names automatically"""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    labels = checkpoint['labels']
    
    # Rebuild Architecture
    model = models.efficientnet_v2_s()
    # Note: Output size is len(labels) - 1 for CORAL
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels)-1)
    
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded! Best Training Accuracy was: {checkpoint['acc']*100:.1f}%")
    return model, labels

def predict_diamond(image_path, model, labels):
    """Takes an image path, returns the Grade"""
    # Simple Preprocessing (same as training validation)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(x)
        # Ordinal Decoding
        probs = torch.sigmoid(logits)
        pred_idx = (probs > 0.5).sum(dim=1).item()
        
    return labels[pred_idx]

# --- Usage Example ---
if __name__ == "__main__":
    model, classes = load_model()
    
    # Test on a random image you have
    test_img = "path/to/your/diamond.jpg" 
    
    # grade = predict_diamond(test_img, model, classes)
    # print(f"Predicted Grade: {grade}")