from pathlib import Path
import argparse
import csv

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dataset"
DESTINATION = ROOT / "experiments" / "classification_ablation" / "masked" / "dataset"
MODEL_PATH = ROOT / "best_segmentation_model_v3.keras"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
SPLITS = ["train", "validation"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_files(split):
    files = []

    for class_name in CLASSES:
        folder = SOURCE / split / class_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                files.append((class_name, path))

    return files


def create_masked_image(model, path):
    with Image.open(path) as source:
        image = source.convert("RGB")

    gray = image.convert("L").resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(gray, dtype=np.float32) / 255.0
    batch = array[np.newaxis, ..., np.newaxis]

    prediction = model.predict(batch, verbose=0)[0, :, :, 0]

    mask = Image.fromarray(
        ((prediction >= 0.5).astype(np.uint8) * 255)
    ).resize(
        image.size,
        Image.Resampling.NEAREST
    )

    image_array = np.asarray(image, dtype=np.uint8).copy()
    mask_array = np.asarray(mask) > 0

    image_array[~mask_array] = 0

    return Image.fromarray(image_array), float(mask_array.mean() * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="0 = tum goruntuler"
    )
    args = parser.parse_args()

    if args.limit_per_class < 0:
        parser.error("--limit-per-class negatif olamaz")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    print("U-Net V3 yukleniyor...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    if tuple(model.input_shape[1:]) != (256, 256, 1):
        raise ValueError(f"Beklenmeyen model girisi: {model.input_shape}")

    pilot = args.limit_per_class > 0

    # Pilot ve tam veri seti ayri tutulur.
    target_root = (
        DESTINATION.parent / "pilot_dataset"
        if pilot else DESTINATION
    )

    records = []

    for split in SPLITS:
        files = get_files(split)

        if pilot:
            selected = []
            for class_name in CLASSES:
                class_files = [
                    item for item in files
                    if item[0] == class_name
                ]
                selected.extend(class_files[:args.limit_per_class])
            files = selected

        print(f"\n{split}: {len(files)} goruntu")

        for index, (class_name, source_path) in enumerate(
            files,
            start=1
        ):
            relative = source_path.relative_to(SOURCE)
            destination = (
                target_root / relative.parent / (relative.stem + ".png")
            )

            if destination.exists():
                print(f"[{index}/{len(files)}] ATLANDI: {relative}")
                continue

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            masked_image, coverage = create_masked_image(
                model,
                source_path
            )

            masked_image.save(destination)

            records.append({
                "split": split,
                "class": class_name,
                "source": str(relative),
                "output": str(destination.relative_to(target_root)),
                "coverage_pct": round(coverage, 2)
            })

            if index % 25 == 0 or index == len(files):
                print(
                    f"[{index}/{len(files)}] "
                    f"{split} - {class_name}"
                )

    target_root.mkdir(parents=True, exist_ok=True)
    report_path = target_root / "generation_report.csv"

    if records:
        with report_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(records[0].keys())
            )
            writer.writeheader()
            writer.writerows(records)

    print("\nIslem tamamlandi.")
    print(f"Cikti: {target_root}")
    print(f"Yeni uretilen goruntu: {len(records)}")


if __name__ == "__main__":
    main()
