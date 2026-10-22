
from pathlib import Path
import csv
import random
import shutil

BASE = Path(__file__).resolve().parents[2]

SOURCE = BASE / "dataset" / "test"
OUTPUT = BASE / "reference_segmentation"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]

SAMPLES_PER_CLASS = 10
SEED = 42

random.seed(SEED)

images_output = OUTPUT / "images"
masks_output = OUTPUT / "masks"

images_output.mkdir(parents=True, exist_ok=True)
masks_output.mkdir(parents=True, exist_ok=True)

manifest_path = OUTPUT / "reference_manifest.csv"

# Var olan orneklemeyi yanlislikla degistirmeyelim.
if manifest_path.exists():
    raise SystemExit(
        "Referans orneklem zaten olusturulmus. "
        "Mevcut reference_manifest.csv korunuyor."
    )

rows = []

for class_name in CLASSES:
    class_folder = SOURCE / class_name

    if not class_folder.is_dir():
        raise FileNotFoundError(class_folder)

    image_files = sorted(
        p for p in class_folder.iterdir()
        if p.suffix.lower() in [
            ".jpg", ".jpeg", ".png"
        ]
    )

    if len(image_files) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"{class_name} icin yeterli goruntu yok."
        )

    selected = random.sample(
        image_files, SAMPLES_PER_CLASS
    )

    output_class = images_output / class_name
    output_class.mkdir(
        parents=True, exist_ok=True
    )

    for index, source_path in enumerate(
        selected, start=1
    ):
        output_name = (
            f"{class_name}_{index:02d}"
            f"{source_path.suffix.lower()}"
        )

        destination = output_class / output_name

        shutil.copy2(
            source_path, destination
        )

        rows.append({
            "class": class_name,
            "original_filename": source_path.name,
            "reference_filename": output_name,
            "mask_status": "PENDING"
        })

        print(
            f"{class_name} {index}/10: "
            f"{source_path.name}",
            flush=True
        )

with manifest_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class",
            "original_filename",
            "reference_filename",
            "mask_status"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nReferans veri grubu hazirlandi.")
print("Toplam goruntu:", len(rows))
print("Klasor:", OUTPUT)
