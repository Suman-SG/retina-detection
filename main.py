import os
import sys
import argparse

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from train import train_model
    from predict import RetinaPredictor
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Current Python path:", sys.path)
    exit(1)

def main():
    parser = argparse.ArgumentParser(description='Retinal Disease Detection')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'predict', 'evaluate'],
                       help='Mode: train, predict, or evaluate')
    parser.add_argument('--image_path', type=str, 
                       help='Path to image for prediction')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        config = {
            'num_classes': 5,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': 0.001,
            'scheduler_step': 5,
            'save_interval': 2,
            'resume_training': True,
            'checkpoint_path': args.checkpoint,
            'train_data_dir': 'data/resized train 19',
            'val_data_dir': 'data/resized train 19'  # We'll split the data
        }
        
        # Create directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        print("🚀 Starting training with APTOS 2019 dataset...")
        train_model(config)
        
    elif args.mode == 'predict':
        if not args.image_path:
            print("❌ Please provide --image_path for prediction")
            return
        
        if not os.path.exists(args.checkpoint):
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            return
        
        if not os.path.exists(args.image_path):
            print(f"❌ Image not found: {args.image_path}")
            return
        
        print("🔮 Starting prediction...")
        predictor = RetinaPredictor(args.checkpoint)
        result = predictor.visualize_prediction(args.image_path)
        
        print(f"\n🎯 Prediction Result:")
        print(f"   Class: {result['class_name']} (Level {result['predicted_class']})")
        print(f"   Confidence: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()