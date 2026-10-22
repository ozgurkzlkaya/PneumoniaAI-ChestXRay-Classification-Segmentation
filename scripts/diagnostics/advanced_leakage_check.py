
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
import hashlib
import json
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "dataset"
OUT = BASE / "final_classification_report"
OUT.mkdir(exist_ok=True)

SPLITS = ["train", "validation", "test"]
CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
EXTS = {".jpg", ".jpeg", ".png"}

# pHash icin DCT tabanli 64 bit imza
from scipy.fft import dctn

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def get_patient_id(filename):
    name = Path(filename).stem.lower()

    match = re.match(r"^(person\d+)_", name)
    if match:
        return match.group(1)

    # NORMAL adlari hasta kimligi olarak dogrulanmadi
    return None

def phash(image):
    gray = image.convert("L").resize(
        (32, 32), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.float32)

    coeff = dctn(pixels, type=2, norm="ortho")
    low = coeff[:8, :8].copy()

    values = low.flatten()[1:]
    median = np.median(values)

    bits = low > median
    return int("".join(
        "1" if x else "0" for x in bits.flatten()
    ), 2)

def normalized_hash(image):
    gray = image.convert("L").resize(
        (256, 256), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(gray, dtype=np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()

records = []

print("\n=== GELISMIS VERI SIZINTISI TESTI ===")

for split in SPLITS:
    for cls in CLASSES:
        folder = DATA / split / cls

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTS
            and not p.name.startswith("._")
        )

        print(f"{split}/{cls}: {len(files)}")

        for path in files:
            try:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image)
                    image.load()

                    records.append({
                        "split": split,
                        "class": cls,
                        "path": str(path.relative_to(DATA)),
                        "patient_id": get_patient_id(path.name),
                        "sha256": sha256_file(path),
                        "normalized_hash": normalized_hash(image),
                        "phash": phash(image),
                    })
            except Exception as e:
                print(f"[HATA] {path}: {e}")
                raise

print(f"\nToplam: {len(records)} goruntu")

def find_exact_matches(key):
    groups = defaultdict(list)

    for row in records:
        if row[key] is not None:
            groups[row[key]].append(row)

    result = []

    for value, items in groups.items():
        if len({r["split"] for r in items}) < 2:
            continue

        for a, b in combinations(items, 2):
            if a["split"] == b["split"]:
                continue

            result.append({
                "type": key,
                "distance": 0,
                "split_a": a["split"],
                "class_a": a["class"],
                "path_a": a["path"],
                "split_b": b["split"],
                "class_b": b["class"],
                "path_b": b["path"],
            })

    return result

exact_file = find_exact_matches("sha256")
exact_normalized = find_exact_matches("normalized_hash")
patient_overlap = find_exact_matches("patient_id")

print("\n=== PHASH BENZERLIK TARAMASI ===")

# 64 bit hash icin Hamming uzakligi <= 5 olan
# bolumler arasi adaylari ara.
#
# 5856 goruntude yaklasik 11 milyon karsilastirma.
# Bu nedenle numpy ile toplu hesaplama yapilir.

phash_pairs = []

for split_a, split_b in combinations(SPLITS, 2):
    left = [r for r in records if r["split"] == split_a]
    right = [r for r in records if r["split"] == split_b]

    print(f"{split_a} <-> {split_b}")

    for a in left:
        for b in right:
            distance = (
                a["phash"] ^ b["phash"]
            ).bit_count()

            if distance <= 5:
                phash_pairs.append({
                    "type": "phash",
                    "distance": distance,
                    "split_a": a["split"],
                    "class_a": a["class"],
                    "path_a": a["path"],
                    "split_b": b["split"],
                    "class_b": b["class"],
                    "path_b": b["path"],
                })

print("\n=== SONUCLAR ===")
print("Birebir ayni dosyalar:", len(exact_file))
print("Normalize edilmis ayni goruntuler:", len(exact_normalized))
print("Ayni taninabilen hasta ID:", len(patient_overlap))
print("pHash yakin eslesmeler (<=5):", len(phash_pairs))

all_results = (
    exact_file
    + exact_normalized
    + patient_overlap
    + phash_pairs
)

df = pd.DataFrame(all_results)

csv_path = OUT / "advanced_leakage_matches.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

summary = {
    "total_images": len(records),
    "split_counts": dict(
        Counter(r["split"] for r in records)
    ),
    "exact_file_matches": len(exact_file),
    "normalized_matches": len(exact_normalized),
    "patient_id_matches": len(patient_overlap),
    "phash_near_matches": len(phash_pairs),
    "phash_threshold": 5,
    "notes": [
        "pHash matches are candidates, not confirmed duplicates.",
        "NORMAL patient identity is not verified.",
        "Filename-derived person IDs require source validation.",
    ],
}

json_path = OUT / "advanced_leakage_summary.json"

json_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

if not df.empty:
    print("\nIlk 20 eslesme:")
    print(df.head(20).to_string(index=False))

print("\nCSV:", csv_path)
print("JSON:", json_path)
print("\nTest tamamlandi.")
