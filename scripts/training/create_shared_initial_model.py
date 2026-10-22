from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT
    / "experiments"
    / "classification_ablation"
    / "configs"
    / "shared_initial_model.keras"
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(
            height_factor=0.03,
            width_factor=0.03
        ),
    ],
    name="data_augmentation"
)

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    3,
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="Pneumonia_MobileNetV2"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

if OUTPUT.exists():
    raise FileExistsError(
        f"Baslangic modeli zaten mevcut: {OUTPUT}"
    )

model.save(OUTPUT)

print("\nORTAK BASLANGIC MODELI HAZIR")
print("Kayit:", OUTPUT)
print("Giris:", model.input_shape)
print("Cikis:", model.output_shape)
print("Parametre:", model.count_params())
