import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import os

# Paths
DATASET_DIR = r'E:\HackOrbit\classified_dataset'  # Update if needed
MODEL_PATH = 'pothole_severity_model.h5'

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# Data generators with strong augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    rotation_range=40,
    zoom_range=0.4,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Print class distribution
print("Class indices:", train_gen.class_indices)
print("Training samples per class:", np.bincount(train_gen.classes))

# Compute class weights for balanced learning
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(train_gen.class_indices)),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Build model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(len(train_gen.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model for initial training
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Train top layers
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS//2,
    class_weight=class_weights
)

# Unfreeze last 40 layers for fine-tuning
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS//2,
    class_weight=class_weights
)

# Save model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Evaluate and print confusion matrix
val_gen.reset()
y_true = val_gen.classes
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(train_gen.class_indices.keys())))