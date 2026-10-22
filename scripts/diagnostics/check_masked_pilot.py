from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path.cwd()
SOURCE = ROOT / "dataset"
PILOT = ROOT / "experiments/classification_ablation/masked/pilot_dataset"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
SPLITS = ["train", "validation"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

issues = []
total = 0

for split in SPLITS:
    print(f"\n{split.upper()}")

    for class_name in CLASSES:
        source_dir = SOURCE / split / class_name
        output_dir = PILOT / split / class_name

        source_files = sorted(
            p for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )[:5]

        expected = [p.stem + ".png" for p in source_files]

        if len(expected) != len(set(expected)):
            issues.append(f"Dosya adi cakismasi: {split}/{class_name}")

        output_files = (
            sorted(output_dir.glob("*.png"))
            if output_dir.exists() else []
        )

        print(
            f"{class_name:10s} "
            f"beklenen={len(expected)} "
            f"bulunan={len(output_files)}"
        )

        actual = {p.name for p in output_files}

        for filename in expected:
            if filename not in actual:
                issues.append(f"Eksik: {split}/{class_name}/{filename}")

        for path in output_files:
            total += 1
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.mode != "RGB":
                        issues.append(f"RGB degil: {path}")
                    if min(image.size) <= 0:
                        issues.append(f"Gecersiz boyut: {path}")
            except Exception as error:
                issues.append(f"Okuma hatasi: {path} - {error}")

print("\nTOPLAM:", total)
print("SORUN SAYISI:", len(issues))

for issue in issues:
    print("SORUN:", issue)
