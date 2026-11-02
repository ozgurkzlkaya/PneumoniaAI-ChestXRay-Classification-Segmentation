from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# AYARLAR
# ============================================================

DATASET_DIR = Path("dataset")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

CLASS_NAMES = ["NORMAL", "BACTERIAL", "VIRAL"]

tf.random.set_seed(SEED)


# ============================================================
# VERI SETLERI
# ============================================================

print("\nVeri setleri yukleniyor...\n")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

# X-Ray goruntulerinde horizontal flip kullanmiyoruz.

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.05),
    layers.RandomTranslation(
        height_factor=0.03,
        width_factor=0.03
    ),
], name="data_augmentation")


# ============================================================
# MOBILENETV2 TABAN MODELI
# ============================================================

print("\nMobileNetV2 yukleniyor...\n")

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Ilk asamada MobileNetV2 agirliklarini degistirmiyoruz.
base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 icin gerekli on isleme
x = keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

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


# ============================================================
# DERLEME
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CLASS WEIGHT
# ============================================================

class_weight = {
    0: 1.231,   # NORMAL
    1: 0.714,   # BACTERIAL
    2: 1.272    # VIRAL
}


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        "best_transfer_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# MODEL OZETI
# ============================================================

print("\n==============================")
print("TRANSFER LEARNING MODEL OZETI")
print("==============================\n")

model.summary()


# ============================================================
# EGITIM
# ============================================================

print("\n==============================")
print("MODEL 2 EGITIMI BASLIYOR")
print("==============================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)


# ============================================================
# KAYDET
# ============================================================

model.save(
    "final_transfer_model.keras"
)

print("\n==============================")
print("EGITIM TAMAMLANDI")
print("==============================")

print("\nEn iyi model:")
print("best_transfer_model.keras")

print("\nFinal model:")
print("final_transfer_model.keras")