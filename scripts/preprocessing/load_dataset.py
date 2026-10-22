from pathlib import Path
import tensorflow as tf

# =========================
# AYARLAR
# =========================

DATASET_DIR = Path("dataset")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Sınıf numaralarını özellikle belirliyoruz:
# 0 = NORMAL
# 1 = BACTERIAL
# 2 = VIRAL
CLASS_NAMES = ["NORMAL", "BACTERIAL", "VIRAL"]


# =========================
# TRAIN VERİSİ
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


# =========================
# VALIDATION VERİSİ
# =========================

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


# =========================
# TEST VERİSİ
# =========================

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


# =========================
# KONTROLLER
# =========================

print("\n==============================")
print("VERI SETI BASARIYLA YUKLENDI")
print("==============================")

print("\nSiniflar:")
for index, class_name in enumerate(CLASS_NAMES):
    print(f"{index} = {class_name}")

print("\nGoruntu boyutu:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

print("\nBatch sayilari:")
print("Train      :", len(train_ds))
print("Validation :", len(val_ds))
print("Test       :", len(test_ds))


# İlk batch'i kontrol et
for images, labels in train_ds.take(1):

    print("\nIlk batch kontrolu:")
    print("Goruntu tensor boyutu:", images.shape)
    print("Etiket tensor boyutu :", labels.shape)

    print("\nIlk 10 etiket:")
    print(labels[:10].numpy())

    print("\nPixel deger araligi:")
    print("Minimum:", tf.reduce_min(images).numpy())
    print("Maximum:", tf.reduce_max(images).numpy())