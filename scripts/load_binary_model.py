import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# --------------------------
# SETTINGS
# --------------------------
MODEL_DIR = "models"
IMG_SIZE = 224
BATCH_SIZE = 32

# --------------------------
# VALIDATION DATASET
# --------------------------
print("📂 Loading validation data...")

val_datagen = ImageDataGenerator(rescale=1/255., validation_split=0.1)

# IMPORTANT: must be same CSV as training
df = None
import pandas as pd
df = pd.read_csv("data/train_binary.csv")
df["binary"] = df["binary"].astype(str)
df["filepath"] = df.apply(lambda r: os.path.join("data", r["source"], r["id_code"]), axis=1)

val_gen = val_datagen.flow_from_dataframe(
    df,
    x_col="filepath",
    y_col="binary",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# --------------------------
# BUILD MODEL ARCHITECTURE
# --------------------------
print("🧱 Building model...")

base = EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(base.input, out)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# --------------------------
# TEST ALL WEIGHT FILES
# --------------------------
print("\n🔍 Checking all epoch weights...\n")

weights_files = sorted(
    [f for f in os.listdir(MODEL_DIR) if f.endswith(".weights.h5")]
)

results = {}

for wf in weights_files:
    path = os.path.join(MODEL_DIR, wf)
    print(f"➡ Loading weights: {wf}")

    model.load_weights(path)

    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"   ✔ accuracy = {acc:.4f}")

    results[wf] = acc

# --------------------------
# SELECT BEST MODEL
# --------------------------
best_file = max(results, key=results.get)
best_acc = results[best_file]

print("\n🎯 BEST MODEL FOUND!")
print(f"📌 File : {best_file}")
print(f"📈 Accuracy : {best_acc:.4f}")

# --------------------------
# SAVE BEST MODEL AS .keras
# --------------------------
print("\n💾 Saving final best model...")

model.load_weights(os.path.join(MODEL_DIR, best_file))
model.save("models/binary_best_model.keras", save_traces=False)

print("🎉 Saved → models/binary_best_model.keras")
