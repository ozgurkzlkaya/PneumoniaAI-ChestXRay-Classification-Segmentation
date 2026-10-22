
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

BASE = Path(__file__).resolve().parents[2]

DATA = (
    BASE / "segmentation_dataset"
    / "prepared_shenzhen_v2"
)

MODEL_PATH = (
    BASE / "best_segmentation_model.keras"
)

OUTPUT_MODEL = (
    BASE / "best_segmentation_model_v2.keras"
)

SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-5

tf.keras.utils.set_random_seed(42)


def load_split(split):
    image_dir = DATA / split / "images"
    mask_dir = DATA / split / "masks"

    images = []
    masks = []

    for image_path in sorted(
        image_dir.glob("*.png")
    ):
        mask_path = (
            mask_dir / image_path.name
        )

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        with Image.open(image_path) as im:
            image = np.asarray(
                im.convert("L"),
                dtype=np.float32
            ) / 255.0

        with Image.open(mask_path) as m:
            mask = np.asarray(
                m.convert("L"),
                dtype=np.float32
            ) / 255.0

        images.append(image[..., None])
        masks.append(mask[..., None])

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(masks, dtype=np.float32)
    )


print("Veriler yukleniyor...", flush=True)

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)

print("Mevcut U-Net yukleniyor...", flush=True)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(
        y_true, y_pred
    )
    return (
        tf.reduce_mean(bce)
        + dice_loss(y_true, y_pred)
    )


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=combined_loss,
    metrics=[
        dice_coef,
        tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy"
        )
    ]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_MODEL,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("\nU-Net V2 egitimi basliyor...", flush=True)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

print("\nEgitim tamamlandi.")
print("Yeni model:", OUTPUT_MODEL)
