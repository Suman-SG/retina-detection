import os
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

CSV_PATH = "data/train_binary.csv"
IMG_SIZE = 224
BATCH_SIZE = 20
EPOCHS = 12

print("\n📄 Loading CSV...")
df = pd.read_csv(CSV_PATH)

# -----------------------
# BALANCE DATASET
# -----------------------
df_0 = df[df["binary"] == 0]
df_1 = df[df["binary"] == 1]

N = min(len(df_0), len(df_1))  # equal count

df = pd.concat([
    df_0.sample(N, random_state=42),
    df_1.sample(N, random_state=42)
])

df = shuffle(df).reset_index(drop=True)

print("\nBalanced counts:")
print(df["binary"].value_counts())

df["binary"] = df["binary"].astype(str)

df["filepath"] = df.apply(
    lambda r: os.path.join("data", r["source"], r["id_code"]),
    axis=1
)

train_gen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.1
)

train = train_gen.flow_from_dataframe(
    df,
    x_col="filepath",
    y_col="binary",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val = train_gen.flow_from_dataframe(
    df,
    x_col="filepath",
    y_col="binary",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

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

print("\n🚀 Training...")
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS
)

os.makedirs("models", exist_ok=True)
model.save("models/binary_balanced.keras", save_traces=False)

print("\n🎉 Saved model → models/binary_balanced.keras")
