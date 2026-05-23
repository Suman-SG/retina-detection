import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from load import load_data  # make sure load_data() returns combined 2015 + 2019 data


# =========================================================
# 1️⃣ Split Data
# =========================================================
def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and validation sets with stratification
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['diagnosis'],
        random_state=random_state
    )

    print(f"📊 Data Split:")
    print(f"  Training set: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")

    # Class distributions
    print("\n📈 Class Distribution in Training:")
    print(train_df['diagnosis'].value_counts().sort_index())
    print("\n📈 Class Distribution in Validation:")
    print(val_df['diagnosis'].value_counts().sort_index())

    return train_df, val_df


# =========================================================
# 2️⃣ Create Data Generators
# =========================================================
def get_data_generators(train_df, val_df, img_dir, image_col='image', target_col='diagnosis',
                       img_size=224, batch_size=32):
    """
    Create ImageDataGenerators for training and validation datasets
    """
    print("\n🌀 Creating advanced data generators...")
    print(f"📊 Training images: {len(train_df)}")
    print(f"📊 Validation images: {len(val_df)}")
    print(f"🎯 Number of classes: {train_df[target_col].nunique()}")

    # ✅ Ensure labels are strings for Keras
    train_df[target_col] = train_df[target_col].astype(str)
    val_df[target_col] = val_df[target_col].astype(str)

    # ✅ Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # ✅ Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)

    # ✅ Training generator
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=img_dir,
        x_col=image_col,
        y_col=target_col,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    # ✅ Validation generator
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=img_dir,
        x_col=image_col,
        y_col=target_col,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    print(f"✅ Training samples: {train_gen.samples}")
    print(f"✅ Validation samples: {val_gen.samples}")
    print(f"✅ Classes mapping: {train_gen.class_indices}")

    return train_gen, val_gen


# =========================================================
# 3️⃣ Compute Class Weights
# =========================================================
def calculate_class_weights(train_df, target_col='diagnosis'):
    """
    Calculate class weights to handle imbalance
    """
    y = train_df[target_col].astype(int).values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = dict(enumerate(class_weights))

    print("\n⚖️ Class Weights (from training data):")
    for cls, weight in class_weight_dict.items():
        train_count = len(train_df[train_df[target_col].astype(int) == cls])
        print(f"  Class {cls}: weight={weight:.2f} | training_samples={train_count}")

    return class_weight_dict


# =========================================================
# 4️⃣ Optional: Test Generator
# =========================================================
def get_test_generator(test_df, img_dir, image_col='image', target_col='diagnosis',
                      img_size=224, batch_size=32):
    """
    Create test data generator (no augmentation)
    """
    test_df[target_col] = test_df[target_col].astype(str)
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=img_dir,
        x_col=image_col,
        y_col=target_col,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print(f"✅ Test samples: {test_gen.samples}")
    return test_gen


# =========================================================
# 5️⃣ Main: Combine everything
# =========================================================
if __name__ == "__main__":
    print("🔍 Loading Retina Datasets (2015 + 2019)...")

    df, paths = load_data()  # ✅ Must return combined df + dictionary of paths

    # Check dataset mix
    print(f"📄 Total combined images: {len(df)}")
    print(f"📊 Combined class distribution:\n{df['diagnosis'].value_counts().sort_index()}")

    # ✅ 80-20 split
    print("\n🎯 Performing 80-20 train-validation split...")
    train_df, val_df = split_data(df, test_size=0.2, random_state=42)

    # ✅ You can switch easily between datasets:
    # Example: use only 2019 for now
    print("\n🔧 Using APTOS 2019 dataset for example training...")
    train_2019 = train_df[train_df['dataset'] == '2019'].copy()
    val_2019 = val_df[val_df['dataset'] == '2019'].copy()

    # ✅ Generate batches
    train_gen, val_gen = get_data_generators(
        train_df=train_2019,
        val_df=val_2019,
        img_dir=paths['train19_imgs'],
        image_col='image',
        target_col='diagnosis'
    )

    # ✅ Compute weights
    class_weights = calculate_class_weights(train_2019)

    print("\n🎯 Data Generators ready for model training!")
    print("🚀 Use train_gen, val_gen, and class_weights in model.fit()")
