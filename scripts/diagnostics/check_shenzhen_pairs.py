
from pathlib import Path
from PIL import Image
import numpy as np

BASE = Path(__file__).resolve().parents[2]

ROOT = (
    BASE / "segmentation_dataset"
    / "Shenzhen" / "Shenzhen"
)

IMG_DIR = ROOT / "img"
MASK_DIR = ROOT / "mask"

EXTENSIONS = {".png", ".jpg", ".jpeg"}

images = sorted(
    p for p in IMG_DIR.iterdir()
    if p.suffix.lower() in EXTENSIONS
)

masks = sorted(
    p for p in MASK_DIR.iterdir()
    if p.suffix.lower() in EXTENSIONS
)

image_map = {p.stem: p for p in images}
mask_map = {p.stem: p for p in masks}

common = sorted(
    set(image_map) & set(mask_map)
)

missing_masks = sorted(
    set(image_map) - set(mask_map)
)

missing_images = sorted(
    set(mask_map) - set(image_map)
)

print("=" * 60)
print("SHENZHEN GORUNTU-MASKE DOGRULAMASI")
print("=" * 60)

print("Goruntu sayisi:", len(images))
print("Maske sayisi:", len(masks))
print("Eslesen dosya adi:", len(common))
print("Maskesi olmayan:", len(missing_masks))
print("Goruntusu olmayan:", len(missing_images))

if missing_masks:
    print("\nMaskesi olmayan ilk 10:")
    print(missing_masks[:10])

if missing_images:
    print("\nGoruntusu olmayan ilk 10:")
    print(missing_images[:10])

size_mismatches = []
empty_masks = []
pixel_values = set()

for name in common:
    with Image.open(image_map[name]) as img:
        image_size = img.size

    with Image.open(mask_map[name]) as mask_img:
        mask_size = mask_img.size
        mask = np.asarray(
            mask_img.convert("L")
        )

    if image_size != mask_size:
        size_mismatches.append(name)

    if np.max(mask) == 0:
        empty_masks.append(name)

    pixel_values.update(
        np.unique(mask).tolist()
    )

print("\nBoyutu uyusmayan:", len(size_mismatches))
print("Tamamen bos maske:", len(empty_masks))
print("Maske piksel degerleri:", sorted(pixel_values)[:30])

print("\nIlk 5 eslesme:")
for name in common[:5]:
    print(
        image_map[name].name,
        "<->",
        mask_map[name].name
    )

print("\nKontrol tamamlandi.")
