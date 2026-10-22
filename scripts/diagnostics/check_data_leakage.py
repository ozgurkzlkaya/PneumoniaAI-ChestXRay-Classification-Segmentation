
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import re

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "dataset"
OUT = BASE / "final_classification_report"
OUT.mkdir(exist_ok=True)

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
EXTS = {".jpg", ".jpeg", ".png"}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def patient_id(path):
    name = path.stem.lower()

    # Pnömoni görüntüleri: person1011_bacteria_2942
    match = re.match(r"^(person\d+)_", name)
    if match:
        return match.group(1)

    # NORMAL görüntülerde IM-0001-0001 gibi adlar olabilir.
    # Bunlar doğrulanmış hasta kimliği değildir.
    return None

records = []

print("\n=== VERI SIZINTISI KONTROLU ===")

for split in SPLITS:
    for class_name in CLASSES:
        folder = DATA / split / class_name

        if not folder.exists():
            print(f"[UYARI] Klasor bulunamadi: {folder}")
            continue

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS
        )

        print(f"{split}/{class_name}: {len(files)} goruntu")

        for path in files:
            records.append({
                "split": split,
                "class": class_name,
                "path": str(path.relative_to(DATA)),
                "filename": path.name.lower(),
                "sha256": sha256_file(path),
                "patient_id": patient_id(path),
            })

if not records:
    raise RuntimeError("Kontrol edilecek goruntu bulunamadi.")

def cross_split_matches(key):
    groups = defaultdict(list)

    for row in records:
        value = row[key]
        if value is not None:
            groups[value].append(row)

    matches = []

    for value, items in groups.items():
        splits = sorted(set(item["split"] for item in items))

        if len(splits) > 1:
            matches.append({
                "value": value,
                "splits": splits,
                "files": [
                    {
                        "split": item["split"],
                        "class": item["class"],
                        "path": item["path"],
                    }
                    for item in items
                ],
            })

    return matches

filename_matches = cross_split_matches("filename")
hash_matches = cross_split_matches("sha256")
patient_matches = cross_split_matches("patient_id")

print("\n=== SONUCLAR ===")
print(f"Toplam goruntu: {len(records)}")
print(f"Farkli bolumlerde ayni dosya adi: {len(filename_matches)}")
print(f"Farkli bolumlerde ayni SHA-256: {len(hash_matches)}")
print(f"Farkli bolumlerde ayni hasta ID: {len(patient_matches)}")

print("\nHasta kimligi kapsamı:")

for split in SPLITS:
    subset = [r for r in records if r["split"] == split]
    known = [r for r in subset if r["patient_id"] is not None]

    print(
        f"{split}: {len(known)}/{len(subset)} "
        "goruntude hasta benzeri kimlik tespit edildi"
    )

for title, matches in [
    ("DOSYA ADI", filename_matches),
    ("SHA256", hash_matches),
    ("HASTA ID", patient_matches),
]:
    if matches:
        print(f"\n[UYARI] {title} ESLESMELERI")

        for match in matches[:10]:
            print(
                match["value"],
                "->",
                match["splits"],
            )

report = {
    "total_images": len(records),
    "split_counts": dict(
        Counter(row["split"] for row in records)
    ),
    "filename_matches": filename_matches,
    "sha256_matches": hash_matches,
    "patient_matches": patient_matches,
    "patient_id_note": (
        "Only personNNN-style filenames were parsed. "
        "NORMAL patient IDs were not verified."
    ),
}

output = OUT / "data_leakage_report.json"
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nRapor kaydedildi:", output)

if hash_matches or patient_matches:
    print(
        "\nSONUC: Olası veri sizintisi tespit edildi. "
        "Eslesmeler incelenmeli."
    )
else:
    print(
        "\nSONUC: Kontrol edilen alanlarda bolumler "
        "arasi eslesme bulunamadi."
    )
    print(
        "Not: Bu, benzer goruntulerin veya "
        "bilinmeyen hasta kimliklerinin "
        "kesinlikle ayrik oldugunu kanitlamaz."
    )
