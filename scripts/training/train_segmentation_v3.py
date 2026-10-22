
from pathlib import Path
import random
import csv

import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# AYARLAR
# ==========================================
SEED = 42
SIZE = 256
BATCH_SIZE = 4
EPOCHS = 25
LEARNING_RATE = 1e-5

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

BASE = Path(__file__).resolve().parents[2]

MONT = BASE / "segmentation_dataset" / "prepared"

SHEN = (
    BASE / "segmentation_dataset"
    / "prepared_shenzhen_v2"
)

V1_MODEL = BASE / "best_segmentation_model.keras"
V3_MODEL = BASE / "best_segmentation_model_v3.keras"

TEST_LIST = BASE / "segmentation_test_files.txt"

# ==========================================
# MONTGOMERY AYRIMINI YENIDEN OLUSTUR
# ==========================================
mont_files = sorted(
    (MONT / "images").glob("*.png")
)

assert len(mont_files) == 138

rng = np.random.default_rng(SEED)
indices = rng.permutation(len(mont_files))

mont_train = [
    mont_files[i] for i in indices[:96]
]

mont_val = [
    mont_files[i] for i in indices[96:117]
]

mont_test = [
    mont_files[i] for i in indices[117:]
]

# Eski test listesini degistirme.
saved_test = {
    line.strip()
    for line in TEST_LIST.read_text(
        encoding="utf-8-sig"
    ).splitlines()
    if line.strip()
}

assert saved_test == {
    p.name for p in mont_test
}, "Montgomery test ayrimi uyusmuyor!"

assert not (
    {p.name for p in mont_train + mont_val}
    & saved_test
)

print("Montgomery:")
print("Train:", len(mont_train))
print("Val:", len(mont_val))
print("Test (kullanilmayacak):", len(mont_test))

# ==========================================
# SHENZHEN AYRIMI
# ==========================================
sh_train = sorted(
    (SHEN / "train" / "images").glob("*.png")
)

sh_val = sorted(
    (SHEN / "val" / "images").glob("*.png")
)

sh_test = sorted(
    (SHEN / "test" / "images").glob("*.png")
)

assert len(sh_train) == 396
assert len(sh_val) == 84
assert len(sh_test) == 86

# Shenzhen split manifest kontrolu
manifest = SHEN / "split_manifest.csv"

with manifest.open(
    newline="", encoding="utf-8-sig"
) as f:
    rows = list(csv.DictReader(f))

for split, files in [
    ("train", sh_train),
    ("val", sh_val),
    ("test", sh_test),
]:
    expected = {
        row["filename"]
        for row in rows
        if row["split"] == split
    }
    assert expected == {
        p.name for p in files
    }, f"Shenzhen {split} manifest uyusmazligi!"

print("\nShenzhen:")
print("Train:", len(sh_train))
print("Val:", len(sh_val))
print("Test (kullanilmayacak):", len(sh_test))

# ==========================================
# GORUNTU VE MASKE YUKLEME
# ==========================================
def load_pair(image_path, mask_path):
    with Image.open(image_path) as img:
        x = np.asarray(
            img.convert("L").resize(
                (SIZE, SIZE),
                Image.Resampling.BILINEAR
            ),
            dtype=np.float32
        ) / 255.0

    with Image.open(mask_path) as img:
        y = np.asarray(
            img.convert("L").resize(
                (SIZE, SIZE),
                Image.Resampling.NEAREST
            ),
            dtype=np.uint8
        )

    y = (y >= 128).astype(np.float32)

    return x[..., None], y[..., None]


def load_mont(paths):
    xs, ys = [], []

    for path in paths:
        x, y = load_pair(
            path,
            MONT / "masks" / path.name
        )
        xs.append(x)
        ys.append(y)

    return np.stack(xs), np.stack(ys)


def load_shenzhen(paths, split):
    xs, ys = [], []

    for path in paths:
        x, y = load_pair(
            path,
            SHEN / split / "masks" / path.name
        )
        xs.append(x)
        ys.append(y)

    return np.stack(xs), np.stack(ys)


print("\nVeriler yukleniyor...", flush=True)

xm_train, ym_train = load_mont(mont_train)
xm_val, ym_val = load_mont(mont_val)

xs_train, ys_train = load_shenzhen(
    sh_train, "train"
)

xs_val, ys_val = load_shenzhen(
    sh_val, "val"
)

# ==========================================
# DENGELI KARMA EGITIM
# ==========================================
# Montgomery: 96 x 4 = 384
# Shenzhen: 396
#
# Dogrulama:
# Montgomery: 21 x 4 = 84
# Shenzhen: 84
#
# Tekrarlar yeni goruntu degildir.
# Yalnizca kaynak dengesini saglar.

x_train = np.concatenate([
    np.tile(xm_train, (4, 1, 1, 1)),
    xs_train
])

y_train = np.concatenate([
    np.tile(ym_train, (4, 1, 1, 1)),
    ys_train
])

x_val = np.concatenate([
    np.tile(xm_val, (4, 1, 1, 1)),
    xs_val
])

y_val = np.concatenate([
    np.tile(ym_val, (4, 1, 1, 1)),
    ys_val
])

print("\nKarma egitim:")
print("Train:", x_train.shape)
print("Val:", x_val.shape)

# ==========================================
# MODEL
# ==========================================
print("\nU-Net V1 yukleniyor...", flush=True)

model = tf.keras.models.load_model(
    V1_MODEL,
    compile=False
)


def dice_coef(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        2.0 * intersection + 1e-6
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + 1e-6
    )


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(
        y_true, y_pred
    )

    return (
        tf.reduce_mean(bce)
        + 1.0 - dice_coef(y_true, y_pred)
    )


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=combined_loss,
    metrics=[dice_coef]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(V3_MODEL),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
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

print("\nU-Net V3 egitimi basliyor...", flush=True)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

print("\nEGITIM TAMAMLANDI")
print("Yeni model:", V3_MODEL)
print(
    "En iyi val_loss:",
    min(history.history["val_loss"])
)
