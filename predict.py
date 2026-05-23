import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model import create_model
from utils import load_checkpoint

class RetinaPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = create_model(num_classes=5, device=self.device)
        
        # Load model without optimizer (for inference only)
        self.model = self.load_model_for_inference(model_path)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        self.severity_descriptions = {
            0: "No apparent diabetic retinopathy detected.",
            1: "Mild non-proliferative diabetic retinopathy detected.",
            2: "Moderate non-proliferative diabetic retinopathy detected.",
            3: "Severe non-proliferative diabetic retinopathy detected.",
            4: "Proliferative diabetic retinopathy detected. Urgent medical attention recommended."
        }
    
    def load_model_for_inference(self, model_path):
        """Load model for inference without optimizer"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
        
        print(f"✅ Model loaded successfully from {model_path}")
        return self.model
    
    def predict(self, image_path):
        # Load and preprocess image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, 1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'description': self.severity_descriptions[predicted_class],
            'all_probabilities': probabilities.cpu().numpy()[0],
            'original_image': original_image
        }
    
    def visualize_prediction(self, image_path, save_path=None):
        result = self.predict(image_path)
        
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(result['original_image'])
        plt.title(f"Retinal Scan")
        plt.axis('off')
        
        # Plot probabilities
        plt.subplot(1, 3, 2)
        y_pos = np.arange(len(self.class_names))
        probabilities = result['all_probabilities']
        
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        bars = plt.barh(y_pos, probabilities, align='center', alpha=0.7, color=colors)
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Probability')
        plt.xlim(0, 1)
        plt.title('Disease Severity Probabilities')
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', ha='left', va='center')
        
        # Prediction result
        plt.subplot(1, 3, 3)
        plt.axis('off')
        result_text = f"""
        PREDICTION RESULT:
        
        Severity: {result['class_name']}
        Confidence: {result['confidence']:.2%}
        
        {result['description']}
        """
        plt.text(0.1, 0.9, result_text, fontsize=12, va='top', linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[result['predicted_class']], alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 Result saved to: {save_path}")
        
        plt.show()
        
        return result

def main():
    predictor = RetinaPredictor('checkpoints/best_model.pth')
    
    # Test with sample images
    test_images = [
        "data/resized test 19/0005cfc8afb6.jpg",
        "data/resized test 19/003f0afdcd15.jpg",
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
            result = predictor.predict(image_path)
            print(f"   Result: {result['class_name']} ({result['confidence']:.2%} confidence)")
            print(f"   Description: {result['description']}")
        else:
            print(f"❌ Image not found: {image_path}")

if __name__ == "__main__":
    main()