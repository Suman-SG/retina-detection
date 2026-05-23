import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

print("🔍 DATA DIAGNOSTIC: Finding the Problem")
print("=" * 50)

# Test 1: Check if images are loading correctly
def test_image_loading():
    print("📸 Testing image loading...")
    df = pd.read_csv("data/labels/trainLabels15.csv", nrows=10)
    df.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)
    df["filepath"] = "data/resized train 15/" + df["id_code"].astype(str) + ".jpg"
    
    for idx, row in df.iterrows():
        if os.path.exists(row['filepath']):
            try:
                img = tf.keras.preprocessing.image.load_img(row['filepath'])
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                print(f"✅ {row['id_code']}: Loaded successfully - Shape: {img_array.shape}")
            except Exception as e:
                print(f"❌ {row['id_code']}: Failed to load - {e}")
        else:
            print(f"❌ {row['id_code']}: File not found")

# Test 2: Check data generator
def test_data_generator():
    print("\n🔄 Testing data generator...")
    df = pd.read_csv("data/labels/trainLabels15.csv", nrows=100)
    df.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)
    df["filepath"] = "data/resized train 15/" + df["id_code"].astype(str) + ".jpg"
    df = df[df["filepath"].apply(lambda x: os.path.exists(x))]
    df["diagnosis"] = df["diagnosis"].astype(str)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_dataframe(df, x_col="filepath", y_col="diagnosis", 
                                           target_size=(160, 160), batch_size=8,
                                           class_mode="categorical", subset="training")
    
    # Check a batch
    batch_x, batch_y = next(train_gen)
    print(f"📊 Batch shape: {batch_x.shape}")
    print(f"📊 Labels shape: {batch_y.shape}")
    print(f"📊 Sample labels: {batch_y[:5]}")
    
    return len(df)

# Test 3: Quick model test
def test_model_potential():
    print("\n🧪 Testing model potential...")
    df = pd.read_csv("data/labels/trainLabels15.csv", nrows=200)
    df.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)
    df["filepath"] = "data/resized train 15/" + df["id_code"].astype(str) + ".jpg"
    df = df[df["filepath"].apply(lambda x: os.path.exists(x))]
    df["diagnosis"] = df["diagnosis"].astype(str)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_dataframe(df, x_col="filepath", y_col="diagnosis", 
                                           target_size=(160, 160), batch_size=16,
                                           class_mode="categorical", subset="training")
    val_gen = datagen.flow_from_dataframe(df, x_col="filepath", y_col="diagnosis",
                                         target_size=(160, 160), batch_size=16,
                                         class_mode="categorical", subset="validation")
    
    # Simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Quick test
    history = model.fit(train_gen, validation_data=val_gen, epochs=5, verbose=1)
    
    final_acc = history.history['val_accuracy'][-1]
    print(f"🎯 5-epoch test accuracy: {final_acc:.2%}")
    
    if final_acc > 0.40:
        print("✅ Data pipeline is working!")
        return True
    else:
        print("❌ Data pipeline has issues!")
        return False

# Run diagnostics
test_image_loading()
sample_count = test_data_generator()
pipeline_ok = test_model_potential()

print(f"\n📊 DIAGNOSTIC SUMMARY:")
print(f"   Samples found: {sample_count}")
print(f"   Pipeline status: {'✅ WORKING' if pipeline_ok else '❌ BROKEN'}")