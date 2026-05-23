import pandas as pd

df = pd.read_csv("data/train.csv")

# Merge all DR into 1 class
df["binary"] = df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)

df.to_csv("data/train_binary.csv", index=False)

print("DONE! Saved → data/train_binary.csv")
print(df["binary"].value_counts())
