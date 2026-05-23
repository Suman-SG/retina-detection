import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

CSV_PATH = "data/train_binary.csv"
IMG_SIZE = 224
BATCH_SIZE = 20
EPOCHS = 15
MAX_IMAGES = 10000  # ---- limit to 10k images for fast training

print("\n📄 Loading CSV...")
df = pd.read_csv(CSV_PATH)

df["binary"] = df["binary"].astype(str)
df["filepath"] = df.apply(
    lambda r: os.path.join("data", r["source"], r["id_code"]),
    axis=1
)

# -------------------------------
# LIMIT TO 10K IMAGES
# -------------------------------
df = df.sample(n=MAX_IMAGES, random_state=42).reset_index(drop=True)

print(f"\n⚡ Using only {len(df)} images for fast training.")

datagen = ImageDataGenerator(
    rescale=1/255.,
    validation_split=0.1
)

train_gen = datagen.flow_from_dataframe(
    df,
    x_col="filepath",
    y_col="binary",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_dataframe(
    df,
    x_col="filepath",
    y_col="binary",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

base = EfficientNetB0(weights="imagenet", include_top=False,
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(base.input, out)

model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

os.makedirs("models", exist_ok=True)

# -------------------------------------------------
# SAVE WEIGHTS EVERY EPOCH SAFELY (NO JSON, NO ERRORS)
# -------------------------------------------------
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/binary_fast_epoch_{epoch:02d}.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
    verbose=1
)

print("\n🚀 Training started...\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print("\n🎉 Training completed!")
print("✅ All epoch weights saved inside: models/")
