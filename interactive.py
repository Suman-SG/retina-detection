import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def interactive_mode():
    print("🩺 Diabetic Retinopathy Detection System")
    print("=" * 50)
    
    # Check model
    if not os.path.exists('checkpoints/best_model.pth'):
        print("❌ Model not found. Please train first.")
        print("Run: python main.py --mode train --epochs 5")
        return
    
    try:
        from predict import RetinaPredictor
        predictor = RetinaPredictor('checkpoints/best_model.pth')
        
        while True:
            print("\n📁 Enter image path (or 'quit' to exit):")
            image_path = input().strip().strip('"').strip("'")
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
                
            if not os.path.exists(image_path):
                print("❌ Image not found. Please check the path.")
                continue
                
            try:
                result = predictor.predict(image_path)
                
                print("\n" + "="*50)
                print(f"📊 RESULT: {result['class_name']}")
                print(f"✅ Confidence: {result['confidence']:.2%}")
                print(f"📝 {result['description']}")
                print("="*50)
                
                # Show quick probabilities
                print("\n📊 Probabilities:")
                class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
                for i, prob in enumerate(result['all_probabilities']):
                    star = "★" if i == result['predicted_class'] else " "
                    print(f"   {star} {class_names[i]}: {prob:.1%}")
                    
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_mode()