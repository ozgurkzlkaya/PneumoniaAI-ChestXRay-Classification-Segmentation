
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[2]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

print("\n=== TEST VERI KUMESI KESFI ===")

candidates = []

for root in [BASE / "dataset", BASE / "data", BASE]:
    if not root.exists():
        continue

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        if path.name.lower() != "test":
            continue

        # Segmentasyon veri kumelerini dahil etme
        relative = str(path.relative_to(BASE)).lower()

        if any(word in relative for word in [
            "segmentation",
            "shenzhen",
            "montgomery",
            "reference",
        ]):
            continue

        images = [
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        if images:
            candidates.append((path, images))

# Tekrarlanan klasorleri kaldir
seen = set()

for path, images in candidates:
    if path in seen:
        continue
    seen.add(path)

    print(f"\nTest klasoru: {path}")
    print(f"Toplam goruntu: {len(images)}")

    folders = Counter(
        str(p.parent.relative_to(path)) for p in images
    )

    print("\nAlt klasorler:")
    for folder, count in sorted(folders.items()):
        print(f"  {folder}: {count}")

    labels = Counter()
    unknown = []

    for image in images:
        name = image.name.upper()
        parent = image.parent.name.upper()

        if parent in {"NORMAL", "BACTERIAL", "VIRAL"}:
            label = parent
        elif parent == "PNEUMONIA":
            if "BACTERIA" in name:
                label = "BACTERIAL"
            elif "VIRUS" in name or "VIRAL" in name:
                label = "VIRAL"
            else:
                label = "UNKNOWN"
        else:
            label = "UNKNOWN"

        labels[label] += 1

        if label == "UNKNOWN":
            unknown.append(str(image))

    print("\nTespit edilen siniflar:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count}")

    if unknown:
        print("\nEtiketi belirlenemeyen ilk 10 dosya:")
        for name in unknown[:10]:
            print(" ", name)

    print("\nOrnek dosyalar:")
    for image in sorted(images)[:5]:
        print(" ", image.relative_to(path))

if not candidates:
    print("\nTest klasoru otomatik bulunamadi.")
    print("Mevcut veri kumesinin klasor yapisini kontrol edin.")
