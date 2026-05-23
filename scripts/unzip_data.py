import os
import zipfile

def unzip_datasets():
    zip_file = r"C:\Users\shonu\Desktop\dataset.zip"  # Use raw string r""
    
    if os.path.exists(zip_file):
        print(f"📂 Found: {zip_file}")
        os.makedirs("data", exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("data/")
        print(f"✅ Extracted: {zip_file}")
        return True
    else:
        print("❌ No dataset zip file found at the specified path.")
        return False

if __name__ == "__main__":
    unzip_datasets()
