import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from torch.optim import Adam

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model import create_model
from utils import RetinaDataset, get_transforms, load_checkpoint

def evaluate_model(model_path, data_dir, csv_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Load model
    model = create_model(num_classes=5, device=device)
    
    # Create dummy optimizer for loading (won't be used for evaluation)
    dummy_optimizer = Adam(model.parameters(), lr=0.001)
    dummy_scheduler = None
    
    model, _, _, _ = load_checkpoint(model, dummy_optimizer, dummy_scheduler, model_path, device)
    model.eval()
    
    # Load data
    print("📂 Loading evaluation data...")
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded CSV with {len(df)} samples")
        
        # Filter for APTOS 2019 and use a subset for evaluation
        aptos_19_df = df[df['source'] == 'resized train 19']
        if len(aptos_19_df) > 0:
            eval_df = aptos_19_df.sample(min(500, len(aptos_19_df)), random_state=42)
        else:
            eval_df = df.sample(min(500, len(df)), random_state=42)
    else:
        # Use directory-based approach
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            eval_df = pd.DataFrame([{
                'id_code': os.path.splitext(f)[0], 
                'diagnosis': np.random.randint(0, 5)  # Mock labels
            } for f in files[:min(200, len(files))]])
        else:
            print("❌ No data directory found")
            return
    
    print(f"🎯 Evaluating on {len(eval_df)} samples...")
    
    _, test_transform = get_transforms()
    dataset = RetinaDataset(eval_df, data_dir, transform=test_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"📦 Processed {batch_idx + 1} batches...")
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"✅ Overall Accuracy: {100 * accuracy:.2f}%")
    print(f"📈 Cohen's Kappa: {kappa:.3f}")
    print(f"🎯 Total Samples: {len(all_labels)}")
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Confusion Matrix - Diabetic Retinopathy Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Severity', fontsize=12)
    plt.xlabel('Predicted Severity', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Class-wise accuracy
    print("\n🎯 Class-wise Performance:")
    for i, class_name in enumerate(class_names):
        class_mask = all_labels == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_preds[class_mask] == all_labels[class_mask])
            print(f"   {class_name}: {100 * class_acc:.2f}% ({np.sum(class_mask)} samples)")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'correct': (all_preds == all_labels).astype(int)
    })
    
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = [prob[i] for prob in all_probs]
    
    results_df.to_csv('evaluation_results.csv', index=False)
    print(f"\n💾 Detailed results saved to: evaluation_results.csv")
    
    return all_preds, all_labels, all_probs

def analyze_predictions():
    """Analyze model predictions in detail"""
    try:
        results_df = pd.read_csv('evaluation_results.csv')
        
        print("\n" + "="*60)
        print("🔍 PREDICTION ANALYSIS")
        print("="*60)
        
        # Most confident correct predictions
        correct_df = results_df[results_df['correct'] == 1]
        if len(correct_df) > 0:
            correct_df['max_prob'] = correct_df[['prob_No DR', 'prob_Mild', 'prob_Moderate', 'prob_Severe', 'prob_Proliferative DR']].max(axis=1)
            top_confident = correct_df.nlargest(5, 'max_prob')
            print("✅ Top 5 Most Confident Correct Predictions:")
            for idx, row in top_confident.iterrows():
                print(f"   True: {row['true_label']}, Pred: {row['predicted_label']}, Confidence: {row['max_prob']:.3f}")
        
        # Analysis of errors
        error_df = results_df[results_df['correct'] == 0]
        if len(error_df) > 0:
            print(f"\n❌ Error Analysis ({len(error_df)} errors):")
            error_matrix = confusion_matrix(error_df['true_label'], error_df['predicted_label'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(error_matrix, annot=True, fmt='d', cmap='Reds',
                        xticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                        yticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
            plt.title('Error Confusion Matrix', fontsize=14, fontweight='bold')
            plt.ylabel('True Severity')
            plt.xlabel('Predicted Severity')
            plt.tight_layout()
            plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

    except Exception as e:
        print(f"⚠️ Could not analyze predictions: {e}")

if __name__ == "__main__":
    # Run evaluation
    preds, labels, probs = evaluate_model('checkpoints/best_model.pth', 'data/resized train 19', 'data/train.csv')
    
    # Analyze predictions
    analyze_predictions()
    
    print("\n🎉 Evaluation completed! Check:")
    print("   - confusion_matrix.png")
    print("   - error_analysis.png") 
    print("   - evaluation_results.csv")