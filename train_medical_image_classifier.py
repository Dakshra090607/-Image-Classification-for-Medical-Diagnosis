import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================================
# CONFIG (Review and Update Paths)
# ================================
# NOTE: Ensure these paths point to your dataset's location!
BASE_DIR = "dataset/chest_xray" 
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")
test_dir = os.path.join(BASE_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ================================
# DATA AUGMENTATION
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# ================================
# ENHANCEMENT: Data Distribution Check (from untitled25.py concepts)
# ================================
class_indices = train_gen.class_indices
class_labels = list(class_indices.keys())
class_counts = {label: count for label, count in zip(class_labels, np.bincount(train_gen.classes))}
total_samples = train_gen.samples

print("\n📊 Training Data Class Distribution:")
for label, count in class_counts.items():
    percentage = (count / total_samples) * 100
    print(f"  - {label}: {count} samples ({percentage:.1f}%)")

# Apply conditional logic (similar to if-else in untitled25.py)
minority_percentage = min(class_counts.values()) / total_samples * 100
if minority_percentage < 10.0:
    print("\n⚠️ WARNING: **Highly Imbalanced Dataset Detected!**")
    print("Consider adjusting loss function (class weights) or oversampling/undersampling.")
else:
    print("\n✅ Dataset balance appears satisfactory for initial training.")

# ================================
# HELPER TO BUILD MODEL
# ================================
def build_model(base_model_class, name):
    base_model = base_model_class(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # freeze base layers for transfer learning

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output, name=name)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ================================
# TRAIN AND SAVE EACH MODEL
# ================================
results = []

for model_class, name in [
    (VGG16, "VGG16"),
    (ResNet50, "ResNet50"),
    (EfficientNetB0, "EfficientNetB0")
]:
    print(f"\n🧠 Training {name}...")
    model = build_model(model_class, name)

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[es],
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"✅ {name} Test Accuracy: {test_acc:.4f}")

    # Save two versions:
    # 1️⃣ Standard version for predictions
    model.save(os.path.join(MODEL_DIR, f"model_{name}.keras"))

    # 2️⃣ Grad-CAM ready version (same, but renamed for clarity)
    model.save(os.path.join(MODEL_DIR, f"model_{name}_gradcam.keras"))

    results.append({
        "Model": name,
        "Accuracy": round(test_acc, 4),
        "Loss": round(test_loss, 4)
    })

# ================================
# SAVE COMPARISON
# ================================
df = pd.DataFrame(results)
df.to_csv(os.path.join(MODEL_DIR, "model_comparison_gradcam.csv"), index=False)

# ================================
# VISUALIZE COMPARISON
# ================================
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="Model", y="Accuracy")
plt.title("Model Accuracy Comparison")
plt.savefig(os.path.join(MODEL_DIR, "model_comparison_chart_gradcam.png"))

print("\n✅ Training Complete!")
print(df)