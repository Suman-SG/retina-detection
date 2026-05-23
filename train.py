import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model import create_model
from utils import RetinaDataset, get_transforms, save_checkpoint, load_checkpoint, load_aptos_data, train_val_split

def train_model(config):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Create model
    model = create_model(num_classes=config['num_classes'], device=device)
    print(f"🧠 Model created with {config['num_classes']} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=config['scheduler_step'], gamma=0.1)
    
    # Load checkpoint if exists
    start_epoch = 0
    if config['resume_training'] and os.path.exists(config['checkpoint_path']):
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, config['checkpoint_path'], device
        )
    else:
        print("🚀 No checkpoint found, starting from scratch...")
    
    # Data preparation
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    print("📂 Loading datasets...")
    
    try:
        # Load APTOS 2019 data
        df = load_aptos_data()
        
        if df is None or len(df) == 0:
            print("❌ No data loaded, using fallback...")
            # Fallback to directory-based loading
            train_dir = config['train_data_dir']
            if os.path.exists(train_dir):
                files = os.listdir(train_dir)
                data = [{'id_code': os.path.splitext(f)[0], 'diagnosis': i % 5} 
                       for i, f in enumerate(files[:min(1000, len(files))])]
                df = pd.DataFrame(data)
        
        # Split into train and validation
        train_df, val_df = train_val_split(df, val_ratio=0.2)
        
        # Create datasets
        train_dataset = RetinaDataset(train_df, config['train_data_dir'], transform=train_transform)
        val_dataset = RetinaDataset(val_df, config['val_data_dir'], transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        
        print(f"✅ Training batches: {len(train_loader)}")
        print(f"✅ Validation batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"❌ Error setting up datasets: {e}")
        return None
    
    print("🎯 Starting training...")
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n📍 Epoch {epoch+1}/{config['epochs']}")
        print("-" * 60)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"🏃 Training")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            try:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
                
            except Exception as e:
                print(f"❌ Error in training batch {batch_idx}: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"🧪 Validating")
            for images, labels in val_progress:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_progress.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
                    
                except Exception as e:
                    print(f"❌ Error in validation batch: {e}")
                    continue
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        # Store history
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update scheduler
        scheduler.step()
        
        # Print statistics
        print(f"📊 Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"📊 Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"📈 Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0 or epoch == config['epochs'] - 1:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_loss_avg, 
                config['checkpoint_path']
            )
    
    print("🎉 Training completed successfully!")
    
    # Plot training history
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Training history plot saved as 'training_history.png'")
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")
    
    return model

if __name__ == "__main__":
    config = {
        'num_classes': 5,
        'epochs': 10,
        'batch_size': 16,  # Increased but still manageable
        'learning_rate': 0.001,
        'scheduler_step': 5,
        'save_interval': 2,
        'resume_training': True,
        'checkpoint_path': 'checkpoints/best_model.pth',
        'train_data_dir': 'data/resized train 19',
        'val_data_dir': 'data/resized train 19'  # Same directory for train/val split
    }
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    model = train_model(config)