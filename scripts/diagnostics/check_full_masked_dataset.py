from pathlib import Path
from collections import defaultdict

ROOT = Path.cwd()
SOURCE = ROOT / "dataset"
TARGET = ROOT / "experiments/classification_ablation/masked/dataset"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
SPLITS = ["train", "validation"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

issues = []
total = 0

for split in SPLITS:
    print(f"\n{split.upper()}")

    for class_name in CLASSES:
        folder = SOURCE / split / class_name
        files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        total += len(files)

        names = defaultdict(list)

        for path in files:
            names[path.stem.lower()].append(path.name)

        collisions = [
            group for group in names.values()
            if len(group) > 1
        ]

        print(
            f"{class_name:10s} "
            f"goruntu={len(files)} "
            f"isim_cakismasi={len(collisions)}"
        )

        for group in collisions:
            issues.append(
                f"{split}/{class_name}: {group}"
            )

existing = (
    list(TARGET.rglob("*.png"))
    if TARGET.exists() else []
)

print("\nTOPLAM KAYNAK:", total)
print("HEDEFTE MEVCUT PNG:", len(existing))
print("ISIM CAKISMASI:", len(issues))

for issue in issues[:20]:
    print("SORUN:", issue)

if issues:
    print("\nUYARI: Isim cakismalari giderilmeden uretime gecmeyin.")
elif existing:
    print("\nUYARI: Hedefte onceki dosyalar var; once incelenmeli.")
else:
    print("\nON KONTROL BASARILI.")
