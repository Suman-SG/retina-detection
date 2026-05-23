import pandas as pd
import os

df15 = pd.read_csv("data/labels/trainLabels15.csv")
df19 = pd.read_csv("data/labels/trainLabels19.csv")

df15.columns = ["id", "level"]
df19.columns = ["id", "level"]

df15["id"] = df15["id"].astype(str)
df19["id"] = df19["id"].astype(str)

df15["path"] = "data/images15/" + df15["id"] + ".jpeg"
df19["path"] = "data/images19/" + df19["id"] + ".png"

df = pd.concat([df15, df19], axis=0)

print("Total Images:", len(df))

print("\n🔍 Checking sample paths:")
for i in range(10):
    p = df.iloc[i]["path"]
    print(p, "->", os.path.exists(p))

missing = df[~df["path"].apply(os.path.exists)]
print("\n❌ Missing image count:", len(missing))

print("\nExample missing paths:")
print(missing.head())
