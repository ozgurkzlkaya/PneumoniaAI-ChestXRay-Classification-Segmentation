from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path.cwd()

SOURCE = ROOT / "dataset"
MASKED = ROOT / "experiments/classification_ablation/masked/dataset"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
SPLITS = ["train", "validation"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

total = 0
issues = []
coverages = []

for split in SPLITS:
    print(f"\n{split.upper()}")

    for class_name in CLASSES:
        source_dir = SOURCE / split / class_name
        masked_dir = MASKED / split / class_name

        originals = sorted(
            p for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        expected_names = {p.stem + ".png" for p in originals}
        actual_names = {
            p.name for p in masked_dir.glob("*.png")
        } if masked_dir.exists() else set()

        for name in expected_names - actual_names:
            issues.append(f"Eksik: {split}/{class_name}/{name}")

        for name in actual_names - expected_names:
            issues.append(f"Fazladan: {split}/{class_name}/{name}")

        for source_path in originals:
            output_path = masked_dir / (source_path.stem + ".png")

            if not output_path.exists():
                continue

            try:
                with Image.open(source_path) as original:
                    original_size = original.size

                with Image.open(output_path) as masked:
                    masked.load()

                    if masked.size != original_size:
                        issues.append(f"Boyut uyusmazligi: {output_path}")

                    if masked.mode != "RGB":
                        issues.append(f"RGB degil: {output_path}")

                    array = np.asarray(masked)

                    coverage = np.any(array > 0, axis=2).mean() * 100
                    coverages.append(coverage)

                    if coverage < 5 or coverage > 70:
                        issues.append(
                            f"Supheli maske alani %{coverage:.1f}: "
                            f"{split}/{class_name}/{output_path.name}"
                        )

            except Exception as error:
                issues.append(f"Okuma hatasi: {output_path}: {error}")

            total += 1

        print(
            f"{class_name:10s} "
            f"kaynak={len(originals)} "
            f"maskeli={len(actual_names)}"
        )

print("\nKONTROL SONUCU")
print("Kontrol edilen:", total)
print("Ortalama gorunur alan: %{:.2f}".format(
    np.mean(coverages) if coverages else 0
))
print("Uyari sayisi:", len(issues))

for issue in issues[:30]:
    print("UYARI:", issue)
