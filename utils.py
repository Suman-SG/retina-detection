import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RetinaDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, is_test=False):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        
        print(f"📁 Dataset initialized with {len(df)} samples from {data_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        
        # Remove file extension if present and get base name
        img_base = os.path.splitext(img_name)[0]
        
        # Try different file extensions
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image = None
        found_path = None
        
        for ext in extensions:
            img_path = os.path.join(self.data_dir, f"{img_base}{ext}")
            if os.path.exists(img_path):
                try:
                    image = cv2.imread(img_path)
                    if image is not None:
                        found_path = img_path
                        break
                except Exception as e:
                    continue
        
        if image is None:
            # Try with original filename
            img_path = os.path.join(self.data_dir, img_name)
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
                found_path = img_path
        
        if image is None:
            # Create a dummy image for missing files (for debugging)
            print(f"⚠️ Image not found: {img_name} in {self.data_dir}")
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if idx < 3:  # Print first 3 files for verification
                print(f"✅ Loaded: {os.path.basename(found_path)}")
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, img_name
        else:
            label = self.df.iloc[idx]['diagnosis']
            return image, torch.tensor(label, dtype=torch.long)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved: {path}")

# def load_checkpoint(model, optimizer, scheduler, path, device):
#     """Load model checkpoint"""
#     if not os.path.exists(path):
#         print(f"❌ Checkpoint not found: {path}")
#         return model, optimizer, scheduler, 0
    
#     checkpoint = torch.load(path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
#     if scheduler and checkpoint['scheduler_state_dict']:
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
#     print(f"📥 Checkpoint loaded: {path}, Epoch: {checkpoint['epoch']}")
#     return model, optimizer, scheduler, checkpoint['epoch']
def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint with better error handling"""
    if not os.path.exists(path):
        print(f"❌ Checkpoint not found: {path}")
        return model, optimizer, scheduler, 0
    
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Only load optimizer state if optimizer is provided and exists
        if (optimizer is not None and 
            'optimizer_state_dict' in checkpoint and 
            checkpoint['optimizer_state_dict'] is not None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Only load scheduler state if scheduler is provided and exists
        if (scheduler is not None and 
            'scheduler_state_dict' in checkpoint and 
            checkpoint['scheduler_state_dict'] is not None):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        print(f"📥 Checkpoint loaded: {path}, Epoch: {epoch}")
        return model, optimizer, scheduler, epoch
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return model, optimizer, scheduler, 0
def load_aptos_data():
    """Load APTOS 2019 data with proper filtering"""
    try:
        # Load the main CSV
        csv_path = "data/train.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"📊 Loaded CSV with {len(df)} rows")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Filter for APTOS 2019 data only
            aptos_19_df = df[df['source'] == 'resized train 19'].copy()
            print(f"🎯 APTOS 2019 samples: {len(aptos_19_df)}")
            
            if len(aptos_19_df) > 0:
                print("Sample of APTOS 2019 data:")
                print(aptos_19_df[['id_code', 'diagnosis']].head())
                return aptos_19_df
            else:
                print("❌ No APTOS 2019 data found in CSV")
        
        # Fallback: create from directory listing
        print("🔄 Creating dataset from directory...")
        train_dir = "data/resized train 19"
        if os.path.exists(train_dir):
            files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            data = []
            for file in files:
                img_base = os.path.splitext(file)[0]
                # Assign random labels for testing (you should have proper labels)
                data.append({'id_code': img_base, 'diagnosis': np.random.randint(0, 5)})
            
            return pd.DataFrame(data)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
    
    return None

def train_val_split(df, val_ratio=0.2):
    """Split data into training and validation sets"""
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=42, stratify=df['diagnosis'])
    print(f"📚 Training samples: {len(train_df)}")
    print(f"📚 Validation samples: {len(val_df)}")
    
    # Show class distribution
    print("📊 Training class distribution:")
    print(train_df['diagnosis'].value_counts().sort_index())
    print("📊 Validation class distribution:")
    print(val_df['diagnosis'].value_counts().sort_index())
    
    return train_df, val_df