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
TEST_DIR = DATASET_DIR / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

# Etiket sıramız sabit
CLASS_NAMES = ["NORMAL", "BACTERIAL", "VIRAL"]

# Tekrarlanabilirlik
SEED = 42
tf.random.set_seed(SEED)


# ============================================================
# VERI SETLERINI YUKLE
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

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# PERFORMANS AYARLARI
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

# Tibbi goruntulerde anatomik yonu korumak icin
# horizontal flip kullanmiyoruz.

data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# CNN MODELI
# ============================================================

model = keras.Sequential(
    [
        layers.Input(shape=(224, 224, 3)),

        # Sadece egitim sirasinda augmentation uygulanir
        data_augmentation,

        # Pixel degerleri 0-255 -> 0-1
        layers.Rescaling(1.0 / 255),

        # Blok 1
        layers.Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Blok 2
        layers.Conv2D(
            64,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Blok 3
        layers.Conv2D(
            128,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Blok 4
        layers.Conv2D(
            256,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Feature map -> tek vektor
        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.40),

        # 3 sinif
        layers.Dense(
            3,
            activation="softmax"
        )
    ]
)


# ============================================================
# MODELI DERLE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CLASS WEIGHT
# ============================================================

# Train:
# NORMAL     = 1089
# BACTERIAL  = 1878
# VIRAL      = 1054

# Bakteriyel sinif daha fazla oldugu icin
# sinif dengesizligini agirliklarla azaltmaya calisiyoruz.

class_weight = {
    0: 1.231,   # NORMAL
    1: 0.714,   # BACTERIAL
    2: 1.272    # VIRAL
}


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # En iyi modeli diske kaydet
    keras.callbacks.ModelCheckpoint(
        "best_pneumonia_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    # Validation performansi iyilesmezse erken durdur
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Ogrenme hizini gerektiğinde dusur
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
print("CNN MODEL OZETI")
print("==============================\n")

model.summary()


# ============================================================
# EGITIM
# ============================================================

print("\n==============================")
print("MODEL EGITIMI BASLIYOR")
print("==============================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)


# ============================================================
# EGITIM SONRASI MODELI KAYDET
# ============================================================

model.save("final_pneumonia_model.keras")

print("\nModel kaydedildi:")
print("final_pneumonia_model.keras")

print("\nEn iyi model:")
print("best_pneumonia_model.keras")

print("\nEgitim tamamlandi.")