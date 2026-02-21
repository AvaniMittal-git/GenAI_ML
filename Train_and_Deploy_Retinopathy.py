import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# 1. CONFIGURATION

DATA_DIR = "CEP_1/Sample/sample"
CSV_PATH = "CEP_1/trainLabels.csv"

IMG_SIZE = 224
BATCH_SIZE = 1
EPOCHS = 20
LR = 1e-3

MODEL_DIR = "exported_model"


# 2. IMAGE PREPROCESSING

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # CLAHE for exposure normalization
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = img / 255.0
    return img


# 3. LOAD DATA

df = pd.read_csv(CSV_PATH)
df["image_path"] = df["image"].apply(lambda x: os.path.join(DATA_DIR, x + ".jpeg"))

images = []
labels = []

print("Loading and preprocessing images...")

for idx, row in df.iterrows():
    img = preprocess_image(row["image_path"])
    images.append(img)
    labels.append(row["level"])

X = np.array(images, dtype=np.float32)
y = tf.keras.utils.to_categorical(labels, num_classes=5)


# 4. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
    stratify=labels
)


# 5. DATA AUGMENTATION

train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=BATCH_SIZE
)

test_generator = test_datagen.flow(
    X_test,
    y_test,
    batch_size=BATCH_SIZE
)


# 6. DISTRIBUTED STRATEGY

strategy = tf.distribute.MirroredStrategy()
print(f"Devices in sync: {strategy.num_replicas_in_sync}")

GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync


# 7. CNN ARCHITECTURE

with strategy.scope():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()


# 8. CHECKPOINT CALLBACK

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# 9. TRAIN MODEL

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)


# 10. EVALUATION

loss, acc = model.evaluate(test_generator)
print(f"Final Test Accuracy: {acc * 100:.2f}%")

# 11. SAVE MODEL (TF SERVING FORMAT)

MODEL_DIR = "exported_model/1"
model.export(MODEL_DIR)

print(f"Model exported successfully to {MODEL_DIR}")

# TensorFlow Serving Deployment

# Step 1 — Install TensorFlow Serving (Docker
# docker pull tensorflow/serving

# Step 2 — Serve Model
# docker run -p 8501:8501 --mount type=bind,source=$(pwd)/exported_model,target=/models/dr_model -e MODEL_NAME=dr_model tensorflow/serving
# http://localhost:8501/v1/models/dr_model # browser testing

# Step 3 — REST API Prediction Example

import requests
import json
import numpy as np

data = img.reshape(1,224,224,3).tolist()

payload = {
    "instances": data
}

response = requests.post(
    "http://localhost:8501/v1/models/dr_model:predict",
    json=payload
)

print(response.json())
