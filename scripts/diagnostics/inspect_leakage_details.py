
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[2]

REPORT = (
    BASE
    / "final_classification_report"
    / "data_leakage_report.json"
)

with open(REPORT, "r", encoding="utf-8") as f:
    report = json.load(f)

print("\n=== BIREBIR AYNI GORUNTULER ===")

for index, match in enumerate(report["sha256_matches"], 1):
    print(f"\nEslesme #{index}")
    print("SHA-256:", match["value"])

    for item in match["files"]:
        print("  Bolum:", item["split"])
        print("  Sinif:", item["class"])
        print("  Dosya:", item["path"])

print("\n=== DOGRULAMA KUMESI ARAMASI ===")

search_roots = [
    BASE / "dataset",
    BASE / "data",
    BASE / "prepared_dataset",
    BASE,
]

seen = set()

for root in search_roots:
    if not root.exists():
        continue

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        if path.name.lower() not in {
            "val", "valid", "validation"
        }:
            continue

        if path in seen:
            continue

        seen.add(path)

        relative = str(path.relative_to(BASE)).lower()

        if any(word in relative for word in [
            "segmentation",
            "shenzhen",
            "montgomery",
            "__macosx",
        ]):
            continue

        image_count = sum(
            1 for p in path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }
            and not p.name.startswith("._")
        )

        print(f"{path} -> {image_count} goruntu")

print("\nInceleme tamamlandi.")
