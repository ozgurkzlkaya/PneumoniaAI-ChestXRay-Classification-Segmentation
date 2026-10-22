
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]

SOURCE = (
    BASE_DIR
    / "segmentation_dataset"
    / "NLM-MontgomeryCXRSet"
    / "MontgomerySet"
)

IMAGE_DIR = SOURCE / "CXR_png"
LEFT_DIR = SOURCE / "ManualMask" / "leftMask"
RIGHT_DIR = SOURCE / "ManualMask" / "rightMask"

OUTPUT = BASE_DIR / "segmentation_dataset" / "prepared"
OUT_IMAGES = OUTPUT / "images"
OUT_MASKS = OUTPUT / "masks"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_MASKS.mkdir(parents=True, exist_ok=True)

image_files = sorted(IMAGE_DIR.glob("*.png"))

if not image_files:
    raise RuntimeError("X-Ray goruntuleri bulunamadi!")

success = 0
missing = []

for image_path in image_files:
    name = image_path.name

    left_path = LEFT_DIR / name
    right_path = RIGHT_DIR / name

    if not left_path.exists() or not right_path.exists():
        missing.append(name)
        continue

    with Image.open(image_path) as img:
        image = img.convert("L")

    with Image.open(left_path) as img:
        left = np.array(img.convert("L"))

    with Image.open(right_path) as img:
        right = np.array(img.convert("L"))

    if left.shape != right.shape:
        raise ValueError(f"Maske boyutlari farkli: {name}")

    if left.shape != (image.height, image.width):
        raise ValueError(f"Goruntu ve maske boyutlari farkli: {name}")

    combined = np.logical_or(left > 0, right > 0)
    combined = (combined.astype(np.uint8) * 255)

    image.save(OUT_IMAGES / name)
    Image.fromarray(combined).save(OUT_MASKS / name)

    success += 1

print("=" * 50)
print("SEGMENTASYON VERI HAZIRLAMA")
print("=" * 50)
print(f"Toplam X-Ray       : {len(image_files)}")
print(f"Hazirlanan eslesme : {success}")
print(f"Eksik maske        : {len(missing)}")
print(f"Cikti klasoru      : {OUTPUT}")

if missing:
    print("Eksik maskeli dosyalar:")
    for name in missing:
        print(" -", name)

if success != len(image_files):
    raise RuntimeError("Tum goruntuler icin maske bulunamadi!")
