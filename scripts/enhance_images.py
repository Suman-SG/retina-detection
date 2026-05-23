import cv2
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

def enhance_retina_image(image_path, output_path):
    """Enhance retina image using CLAHE and normalization"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge back
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Brightness normalization
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    # Save enhanced image
    cv2.imwrite(output_path, enhanced)
    return True

def process_dataset():
    """Process all training images"""
    print("🔄 Starting image enhancement...")
    
    # Create enhanced directory
    enhanced_dir = "data/enhanced_images"
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # Load datasets
    df15 = pd.read_csv("data/labels/trainLabels15.csv")
    df19 = pd.read_csv("data/labels/trainLabels19.csv")
    
    df15.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)
    df19.rename(columns={"id_code": "id_code", "diagnosis": "diagnosis"}, inplace=True)
    
    # Process 2015 dataset
    print("Processing 2015 dataset...")
    for idx, row in tqdm(df15.iterrows(), total=len(df15)):
        img_name = f"{row['id_code']}.jpg"
        input_path = f"data/resized train 15/{img_name}"
        output_path = f"{enhanced_dir}/{img_name}"
        
        if os.path.exists(input_path):
            enhance_retina_image(input_path, output_path)
    
    # Process 2019 dataset  
    print("Processing 2019 dataset...")
    for idx, row in tqdm(df19.iterrows(), total=len(df19)):
        img_name = f"{row['id_code']}.jpg"
        input_path = f"data/resized train 19/{img_name}"
        output_path = f"{enhanced_dir}/{img_name}"
        
        if os.path.exists(input_path):
            enhance_retina_image(input_path, output_path)
    
    print("✅ Image enhancement complete!")

if __name__ == "__main__":
    process_dataset()