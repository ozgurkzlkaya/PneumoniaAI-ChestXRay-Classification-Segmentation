
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np
import random
import csv

BASE = Path(__file__).resolve().parents[2]

SOURCE = (
    BASE / "segmentation_dataset"
    / "Shenzhen" / "Shenzhen"
)

OUTPUT = (
    BASE / "segmentation_dataset"
    / "prepared_shenzhen_v2"
)

IMAGE_DIR = SOURCE / "img"
MASK_DIR = SOURCE / "mask"

SEED = 42
SIZE = (256, 256)

random.seed(SEED)

files = sorted(IMAGE_DIR.glob("*.png"))

# Dosya adindaki sayisal kimligi grup olarak al.
groups = defaultdict(list)

for path in files:
    parts = path.stem.split("_")
    group_id = "_".join(parts[:2])
    groups[group_id].append(path)

group_ids = sorted(groups.keys())
random.shuffle(group_ids)

n = len(group_ids)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

split_groups = {
    "train": group_ids[:n_train],
    "val": group_ids[n_train:n_train + n_val],
    "test": group_ids[n_train + n_val:]
}

manifest = []

for split, ids in split_groups.items():
    image_out = OUTPUT / split / "images"
    mask_out = OUTPUT / split / "masks"

    image_out.mkdir(
        parents=True, exist_ok=True
    )
    mask_out.mkdir(
        parents=True, exist_ok=True
    )

    count = 0

    for group_id in ids:
        for image_path in groups[group_id]:
            mask_path = (
                MASK_DIR / image_path.name
            )

            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

            with Image.open(image_path) as im:
                image = im.convert("L").resize(
                    SIZE, Image.Resampling.BILINEAR
                )

            with Image.open(mask_path) as m:
                mask = m.convert("L").resize(
                    SIZE, Image.Resampling.NEAREST
                )

            mask_array = np.asarray(mask)
            binary_mask = (
                mask_array >= 128
            ).astype(np.uint8) * 255

            image.save(
                image_out / image_path.name
            )

            Image.fromarray(
                binary_mask
            ).save(mask_out / image_path.name)

            manifest.append({
                "filename": image_path.name,
                "group_id": group_id,
                "split": split
            })

            count += 1

    print(
        f"{split}: {count} goruntu",
        flush=True
    )

manifest_path = OUTPUT / "split_manifest.csv"

with manifest_path.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename", "group_id", "split"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest)

print("\nVeri hazirlama tamamlandi.")
print("Toplam:", len(manifest))
print("Cikti:", OUTPUT)
