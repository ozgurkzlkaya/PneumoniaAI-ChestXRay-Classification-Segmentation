
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------
# AYARLAR
# ---------------------------
SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "segmentation_dataset" / "prepared"

IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

MODEL_PATH = BASE_DIR / "best_segmentation_model.keras"

# ---------------------------
# VERI KONTROLU VE AYIRMA
# ---------------------------
image_paths = sorted(IMAGE_DIR.glob("*.png"))

if len(image_paths) != 138:
    raise RuntimeError(
        f"138 goruntu bekleniyordu, bulunan: {len(image_paths)}"
    )

for path in image_paths:
    if not (MASK_DIR / path.name).exists():
        raise FileNotFoundError(f"Maske eksik: {path.name}")

rng = np.random.default_rng(SEED)
indices = rng.permutation(len(image_paths))

train_paths = [image_paths[i] for i in indices[:96]]
val_paths = [image_paths[i] for i in indices[96:117]]
test_paths = [image_paths[i] for i in indices[117:]]

print("Train:", len(train_paths))
print("Validation:", len(val_paths))
print("Test:", len(test_paths))

# Test dosyalarini sakla.
split_file = BASE_DIR / "segmentation_test_files.txt"
split_file.write_text(
    "\n".join(p.name for p in test_paths),
    encoding="utf-8"
)

# ---------------------------
# GORUNTU VE MASKE YUKLEME
# ---------------------------
def load_pair(path):
    with Image.open(path) as img:
        img = img.convert("L").resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.BILINEAR
        )
        x = np.asarray(img, dtype=np.float32) / 255.0

    mask_path = MASK_DIR / path.name

    with Image.open(mask_path) as img:
        img = img.convert("L").resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.NEAREST
        )
        y = (np.asarray(img) > 127).astype(np.float32)

    return x[..., None], y[..., None]


def make_arrays(paths):
    xs, ys = [], []

    for path in paths:
        x, y = load_pair(path)
        xs.append(x)
        ys.append(y)

    return np.stack(xs), np.stack(ys)


print("Goruntuler yukleniyor...")

x_train, y_train = make_arrays(train_paths)
x_val, y_val = make_arrays(val_paths)

print("Train shape:", x_train.shape)
print("Val shape:", x_val.shape)

# ---------------------------
# DICE VE IOU
# ---------------------------
@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true * y_pred)
    return (
        2.0 * intersection + 1e-6
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + 1e-6
    )


@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true * y_pred)
    union = (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        - intersection
    )

    return (intersection + 1e-6) / (union + 1e-6)


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(
        y_true, y_pred
    )
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


# ---------------------------
# U-NET MODELI
# ---------------------------
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", activation="relu"
    )(x)
    return x


def build_unet():
    inputs = tf.keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 1)
    )

    c1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 128)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    bridge = conv_block(p4, 256)

    u4 = tf.keras.layers.UpSampling2D()(bridge)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 128)

    u3 = tf.keras.layers.UpSampling2D()(c5)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 64)

    u2 = tf.keras.layers.UpSampling2D()(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 32)

    u1 = tf.keras.layers.UpSampling2D()(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 16)

    outputs = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid"
    )(c8)

    return tf.keras.Model(inputs, outputs)


model = build_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss=combined_loss,
    metrics=[dice_coefficient, iou_metric]
)

model.summary()

# ---------------------------
# EGITIM
# ---------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print("\nU-Net egitimi basliyor...")

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\nEgitim tamamlandi.")
print("Kaydedilen model:", MODEL_PATH)
print("Test dosya listesi:", split_file)
