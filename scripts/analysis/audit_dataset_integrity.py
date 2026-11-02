"""Read-only dataset integrity audit for PneumoniaAI.

Run from project root:
    python audit_dataset_integrity.py --root .

Creates CSV/JSON reports only, never changes dataset images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "validation", "test")
CLASSES = ("NORMAL", "BACTERIAL", "VIRAL")
EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}
PATIENT = re.compile(r"person(\d+)", re.IGNORECASE)
NORMAL = re.compile(r"^(?:(NORMAL\d+)-)?IM-(\d+)(?:-(\d+))?(?:-(\d+))?$", re.IGNORECASE)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identifiers(path: Path, label: str):
    stem = path.stem
    patient = PATIENT.search(stem)
    if patient:
        return "person", f"person:{int(patient.group(1))}", f"person:{int(patient.group(1))}", "person filename token"
    if label == "NORMAL":
        match = NORMAL.fullmatch(stem)
        if match:
            series = (match.group(1) or "IM").upper()
            primary = match.group(2)
            shot = match.group(3) or ""
            # Namespace IM and NORMAL2 separately: numeric IDs alone do not prove patient identity.
            patient_key = f"{series}:{primary}"
            capture_key = f"{series}:{primary}:{shot}" if shot else ""
            return series, patient_key, capture_key, "filename-only candidate; not confirmed patient ID"
    return "unknown", "", "", "unparsed"


def write_csv(path: Path, fieldnames, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    dataset = root / "dataset"
    for split in SPLITS:
        for label in CLASSES:
            if not (dataset / split / label).is_dir():
                parser.error(f"Missing dataset directory: {dataset / split / label}")
    out = (args.out.resolve() if args.out else root / "experiments" / "dataset_integrity_audit" / datetime.now(timezone.utc).strftime("audit_%Y%m%d_%H%M%S_%f"))
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    out.mkdir(parents=True)

    records = []
    counts = Counter()
    for split in SPLITS:
        for label in CLASSES:
            for path in sorted((dataset / split / label).rglob("*")):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                kind, patient_key, capture_key, note = identifiers(path, label)
                record = {
                    "split": split, "class": label,
                    "relative_path": path.relative_to(root).as_posix(),
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_hash(path),
                    "id_namespace": kind,
                    "candidate_group_key": patient_key,
                    "candidate_capture_key": capture_key,
                    "identity_note": note,
                }
                records.append(record)
                counts[(split, label)] += 1

    by_hash = defaultdict(list)
    by_group = defaultdict(list)
    by_capture = defaultdict(list)
    by_unscoped_normal = defaultdict(list)
    for rec in records:
        by_hash[rec["sha256"]].append(rec)
        if rec["candidate_group_key"]:
            by_group[rec["candidate_group_key"]].append(rec)
        if rec["candidate_capture_key"]:
            by_capture[rec["candidate_capture_key"]].append(rec)
        if rec["class"] == "NORMAL":
            m = NORMAL.fullmatch(Path(rec["filename"]).stem)
            if m:
                by_unscoped_normal[m.group(2)].append(rec)

    def cross_split_groups(index):
        return {k: v for k, v in index.items() if len({r["split"] for r in v}) > 1}

    duplicates = cross_split_groups(by_hash)
    candidate_groups = cross_split_groups(by_group)
    candidate_captures = cross_split_groups(by_capture)
    unscoped = cross_split_groups(by_unscoped_normal)

    def expand(groups, field):
        result = []
        for key, items in sorted(groups.items()):
            splits = ",".join(sorted({r["split"] for r in items}))
            for rec in items:
                result.append({"group_key": key, "splits": splits, **rec})
        return result

    fields = list(records[0]) if records else []
    write_csv(out / "inventory.csv", fields, records)
    group_fields = ["group_key", "splits"] + fields
    write_csv(out / "exact_duplicates_cross_split.csv", group_fields, expand(duplicates, "sha256"))
    write_csv(out / "candidate_identity_cross_split.csv", group_fields, expand(candidate_groups, "candidate_group_key"))
    write_csv(out / "candidate_capture_cross_split.csv", group_fields, expand(candidate_captures, "candidate_capture_key"))
    write_csv(out / "normal_numeric_id_cross_namespace.csv", group_fields, expand(unscoped, "numeric"))

    # Same-class and cross-class hash collisions within one split are useful quality checks too.
    within = {k: v for k, v in by_hash.items() if len(v) > 1 and len({r["split"] for r in v}) == 1}
    write_csv(out / "exact_duplicates_within_split.csv", group_fields, expand(within, "sha256"))

    summary = {
        "dataset": str(dataset),
        "total_images": len(records),
        "counts": {split: {label: counts[(split, label)] for label in CLASSES} for split in SPLITS},
        "unparsed_images": sum(not r["candidate_group_key"] for r in records),
        "exact_duplicate_cross_split_groups": len(duplicates),
        "exact_duplicate_within_split_groups": len(within),
        "candidate_identity_cross_split_groups": len(candidate_groups),
        "candidate_capture_cross_split_groups": len(candidate_captures),
        "normal_numeric_id_cross_split_groups_ignoring_namespace": len(unscoped),
        "warnings": [
            "IM and NORMAL2-IM numeric identifiers are NOT assumed to identify the same patient.",
            "Filename-derived group identifiers are candidate matches, not verified patient identities.",
            "SHA-256 finds byte-identical files only, not resized or re-encoded duplicates.",
            "This script does not modify, move, delete or repartition images.",
        ],
    }
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nREPORTS:", out)


if __name__ == "__main__":
    main()
