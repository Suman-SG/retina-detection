import os
import pandas as pd

def check_actual_files():
    base_dir = "data"
    
    # Check APTOS 2019 directories
    train_dir_19 = "data/resized train 19"
    test_dir_19 = "data/resized test 19"
    
    print("📁 Checking APTOS 2019 directories:")
    
    if os.path.exists(train_dir_19):
        files = os.listdir(train_dir_19)
        print(f"Train 19: {len(files)} files")
        if files:
            print(f"First 5 files: {files[:5]}")
    
    if os.path.exists(test_dir_19):
        files = os.listdir(test_dir_19)
        print(f"Test 19: {len(files)} files")
        if files:
            print(f"First 5 files: {files[:5]}")
    
    # Check for CSV files
    print("\n📊 Looking for CSV files:")
    for file in os.listdir(base_dir):
        if file.endswith('.csv'):
            print(f"Found: {file}")
            try:
                df = pd.read_csv(os.path.join(base_dir, file))
                print(f"  Columns: {list(df.columns)}")
                print(f"  Shape: {df.shape}")
                print(f"  First few rows:")
                print(df.head(3))
            except Exception as e:
                print(f"  Error reading: {e}")

if __name__ == "__main__":
    check_actual_files()