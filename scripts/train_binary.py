import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ----------------------------
# SETTINGS
# ----------------------------
CSV_PATH = "data/train_binary.csv"
IMG_SIZE = 224
BATCH_SIZE = 20
EPOCHS = 20

# ----------------------------
# LOAD CSV
# ----------------------------
print("\n📄 Loading binary CSV...")
df = pd.read_csv(CSV_PATH)
print(df["binary"].value_counts())

df["binary"] = df["binary"].astype(str)

# ----------------------------
# BUILD FULL FILE PATH
# ----------------------------
df["filepath"] = df.apply(
    lambda r: os.path.join("data", r["source"], r["id_code"]),
    axis=1
)

print("\nSample entries:")
print(df.head())

# ----------------------------
# DATA GENERATOR
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1/255.,
    validation_split=0.1
)

train_gen = train_datagen.flow_from_dataframe(
    df,
    x_col="filepath",
    y_col="binary",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_dataframe(
    df,
    x_col="filepath",
    y_col="binary",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# ----------------------------
# BUILD MODEL
# ----------------------------
base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(base.input, out)

model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
print("\n🚀 Training started...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/binary_dr_detector.keras", save_traces=False)

print("\n🎉 Model saved: models/binary_dr_detector.keras")
