import tensorflow as tf
import numpy as np
import cv2
import sys
import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

IMG_SIZE = 224

# ---------------------------------------------------
# BUILD SAME MODEL ARCHITECTURE
# ---------------------------------------------------
def build_model():
    base = EfficientNetB0(include_top=False, weights=None,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(base.input, out)
    return model

# ---------------------------------------------------
# LOAD WEIGHTS
# ---------------------------------------------------
WEIGHTS_PATH = "models/binary_fast_epoch_04.weights.h5"   # ← your best model

model = build_model()
model.load_weights(WEIGHTS_PATH)
print("✔ Loaded weights:", WEIGHTS_PATH)

# ---------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred >= 0.5:
        print(f"🩺 DISEASE DETECTED — score: {pred:.4f}")
    else:
        print(f"✅ NORMAL — score: {pred:.4f}")

# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if len(sys.argv) != 2:
    print("Usage: python detect_ext.py <image_path>")
    sys.exit()

predict_image(sys.argv[1])
