import pandas as pd
import os

files = [
    r"data/labels/trainLabels15.csv",
    r"data/labels/trainLabels19.csv",
    r"data/labels/testLabels15.csv",
    r"data/labels/testImages19.csv"
]

print("\n================ DATASET INSPECTOR ================\n")

for f in files:
    print(f"\n📄 FILE:", f)
    
    if not os.path.exists(f):
        print("❌ File NOT FOUND!")
        continue

    try:
        df = pd.read_csv(f)
        print("✅ File Loaded Successfully")
        
        print("➡ Columns:", list(df.columns))
        print("➡ Shape (rows, cols):", df.shape)
        print("➡ Missing values:", df.isnull().sum().to_dict())
        
        print("\n🔍 First 5 Rows:")
        print(df.head())
        print("\n" + "-"*60)

    except Exception as e:
        print("❌ ERROR reading file:", e)

print("\n================ END OF REPORT ================")
