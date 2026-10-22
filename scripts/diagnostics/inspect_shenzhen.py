
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / "segmentation_dataset" / "Shenzhen"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

print("=" * 65)
print("SHENZHEN VERI SETI KONTROLU")
print("=" * 65)

if not ROOT.exists():
    raise FileNotFoundError(
        f"Klasor bulunamadi: {ROOT}"
    )

print(f"\nAna klasor: {ROOT}")

print("\nKLASOR YAPISI")
print("-" * 65)

for path in sorted(ROOT.rglob("*")):
    if path.is_dir():
        relative = path.relative_to(ROOT)
        depth = len(relative.parts)
        if depth <= 4:
            print("  " * depth + f"[{path.name}]")

images = [
    p for p in ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
]

print("\nDOSYA ISTATISTIKLERI")
print("-" * 65)

print(f"Toplam goruntu dosyasi: {len(images)}")

folder_counts = Counter(
    str(p.parent.relative_to(ROOT))
    for p in images
)

for folder, count in sorted(folder_counts.items()):
    print(f"{folder}: {count}")

print("\nORNEK DOSYA YOLLARI")
print("-" * 65)

for path in sorted(images)[:25]:
    print(path.relative_to(ROOT))

print("\nOLASI MASKE KLASORLERI")
print("-" * 65)

mask_keywords = [
    "mask", "label", "ground",
    "manual", "segmentation"
]

candidates = [
    p for p in ROOT.rglob("*")
    if p.is_dir()
    and any(
        word in p.name.lower()
        for word in mask_keywords
    )
]

if candidates:
    for path in candidates:
        print(path.relative_to(ROOT))
else:
    print("Isminden taninan maske klasoru bulunamadi.")

print("\nKontrol tamamlandi.")
