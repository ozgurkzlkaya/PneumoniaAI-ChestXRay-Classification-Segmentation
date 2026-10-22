from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# ============================================================
# AYARLAR
# ============================================================

DATASET_DIR = Path("dataset")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
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
# MODEL 2'YI YUKLE
# ============================================================

print("\nEn iyi MobileNetV2 modeli yukleniyor...\n")

model = keras.models.load_model(
    "best_transfer_model.keras"
)


# ============================================================
# MOBILENETV2 TABAN MODELINI BUL
# ============================================================

base_model = model.get_layer(
    "mobilenetv2_1.00_224"
)

# Once tum taban modeli ac
base_model.trainable = True

# Sadece son 30 katmani egitilebilir yap.
# Onceki katmanlar dondurulmus kalacak.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# BatchNormalization katmanlarini dondurmak
# fine-tuning stabilitesi acisindan daha guvenli.
for layer in base_model.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False


# ============================================================
# KONTROL
# ============================================================

trainable_layers = sum(
    1 for layer in base_model.layers if layer.trainable
)

frozen_layers = sum(
    1 for layer in base_model.layers if not layer.trainable
)

print("\n==============================")
print("FINE-TUNING AYARLARI")
print("==============================")

print("Trainable base layers:", trainable_layers)
print("Frozen base layers   :", frozen_layers)


# ============================================================
# YENIDEN DERLE
# ============================================================

# Fine-tuning icin cok dusuk learning rate.

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
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
        "best_finetuned_model.keras",
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
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# MODEL OZETI
# ============================================================

print("\n==============================")
print("FINE-TUNING MODEL OZETI")
print("==============================\n")

model.summary()


# ============================================================
# FINE-TUNING
# ============================================================

print("\n==============================")
print("FINE-TUNING BASLIYOR")
print("==============================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)


# ============================================================
# FINAL MODELI KAYDET
# ============================================================

model.save(
    "final_finetuned_model.keras"
)

print("\n==============================")
print("FINE-TUNING TAMAMLANDI")
print("==============================")

print("\nEn iyi model:")
print("best_finetuned_model.keras")

print("\nFinal model:")
print("final_finetuned_model.keras")