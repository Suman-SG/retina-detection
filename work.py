import os
import sys
import argparse

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def simple_predict(image_path):
    """Simple prediction without complex dependencies"""
    try:
        from predict import RetinaPredictor
        
        # Check if model exists
        if not os.path.exists('checkpoints/best_model.pth'):
            print("❌ Model not found. Please train the model first.")
            print("Run: python main.py --mode train --epochs 5")
            return
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print("🔄 Loading model...")
        predictor = RetinaPredictor('checkpoints/best_model.pth')
        
        print(f"🔍 Analyzing: {os.path.basename(image_path)}")
        result = predictor.predict(image_path)
        
        # Display result
        print("\n" + "="*50)
        print("🎯 DIABETIC RETINOPATHY DETECTION RESULT")
        print("="*50)
        print(f"📊 Severity Level: {result['class_name']}")
        print(f"✅ Confidence: {result['confidence']:.2%}")
        print(f"📝 {result['description']}")
        print("="*50)
        
        # Show probabilities
        print("\n📈 Probability Distribution:")
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        for i, (class_name, prob) in enumerate(zip(class_names, result['all_probabilities'])):
            indicator = "🎯" if i == result['predicted_class'] else "  "
            print(f"   {indicator} {class_name}: {prob:.1%}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all files are in the same directory.")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection')
    parser.add_argument('image_path', type=str, help='Path to retinal image')
    
    args = parser.parse_args()
    simple_predict(args.image_path)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python work.py <image_path>")
        print("Example: python work.py \"data/resized test 19/0005cfc8afb6.jpg\"")
    else:
        main()