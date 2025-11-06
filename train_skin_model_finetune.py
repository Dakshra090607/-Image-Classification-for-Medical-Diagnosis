import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np 

# --------------------------
# Paths (Updated MODEL_DIR added)
# --------------------------
DATA_DIR = r"C:\Ai_project\dataset\skin"
# ⭐ NEW: Define the dedicated model save directory
MODEL_DIR = r"C:\Ai_project\saved_models" 
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure the directory exists

IMAGE_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2")
]
CSV_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# --------------------------
# Load metadata
# --------------------------
df = pd.read_csv(CSV_PATH)

# Map image_id to actual path
def find_image_path(image_id):
    for folder in IMAGE_DIRS:
        path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

df['image_path'] = df['image_id'].apply(find_image_path)
df = df.dropna(subset=['image_path'])

# ==================================================
# Data Analysis, Conditional Logic, and Weight Calculation
# ==================================================

print("\n--- 📊 Class Distribution Analysis ---")
class_counts = df['dx'].value_counts()
print(class_counts)

# Conditional Logic (Imbalance Check)
min_count = class_counts.min()
total_count = class_counts.sum()
if (min_count / total_count) < 0.05:
    print("\n⚠️ WARNING: Severe Class Imbalance Detected!")
    print("The smallest class makes up only {:.2f}% of the total data.".format((min_count / total_count) * 100))
    print("Applying class weights to mitigate bias.")
else:
    print("\n✅ Class distribution is acceptable. No severe imbalance detected.")


# Calculate Class Weights
class_labels = df['dx'].unique()
class_labels.sort()
class_indices = {name: i for i, name in enumerate(class_labels)}

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=class_labels,
    y=df['dx']
)

# Convert the array to a dictionary required by Keras
class_weights_dict = dict(enumerate(class_weights))

print("\nCalculated Class Weights (Passed to Model.fit):")
print({name: round(class_weights_dict[i], 2) for name, i in class_indices.items()})

# --------------------------
# Train/Validation split
# --------------------------
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['dx'], random_state=42
)

# --------------------------
# Image data generators
# --------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ADVANCED AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    width_shift_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='dx',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='dx',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --------------------------
# Build model (EfficientNetB0)
# --------------------------
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3))
base_model.trainable = False  # freeze base

num_classes = len(class_labels)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --------------------------
# Step 1: Train top layers (Applied Class Weights)
# --------------------------
EPOCHS_TOP = 10
print("\n--- Training Top Layers (Stage 1) ---")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_TOP,
    class_weight=class_weights_dict
)

# --------------------------
# Step 2: Fine-tune some layers (Applied Class Weights)
# --------------------------
# Unfreeze last 50 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR for fine-tuning
              loss='categorical_crossentropy', metrics=['accuracy'])

EPOCHS_FINE = 10
print("\n--- Fine-Tuning Base Model (Stage 2) ---")
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    class_weight=class_weights_dict
)

# --------------------------
# Save model (Path updated to use MODEL_DIR)
# --------------------------
# ⭐ UPDATED SAVE PATH ⭐
model_save_path = os.path.join(MODEL_DIR, "skin_cancer_model_finetuned.h5")
model.save(model_save_path)
print(f"\nFine-tuned model saved at: {model_save_path}")