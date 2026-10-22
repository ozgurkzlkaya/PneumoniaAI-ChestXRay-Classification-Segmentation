
from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
from PIL import Image

BASE = Path(__file__).resolve().parents[2]

MODELS = {
    "V1": BASE / "best_segmentation_model.keras",
    "V2": BASE / "best_segmentation_model_v2.keras",
    "V3": BASE / "best_segmentation_model_v3.keras"
}


MONT = BASE / "segmentation_dataset" / "prepared"
SHEN = (
    BASE / "segmentation_dataset"
    / "prepared_shenzhen_v2" / "test"
)

TEST_FILE = BASE / "segmentation_test_files.txt"

OUT = BASE / "results" / "segmentation" / "segmentation_comparison_v1_v2_v3"
OUT.mkdir(exist_ok=True)

SIZE = (256, 256)
THRESHOLD = 0.5


def load_pair(img_path, mask_path):
    with Image.open(img_path) as im:
        x = np.asarray(
            im.convert("L").resize(
                SIZE, Image.Resampling.BILINEAR
            ), dtype=np.float32
        ) / 255.0

    with Image.open(mask_path) as im:
        y = np.asarray(
            im.convert("L").resize(
                SIZE, Image.Resampling.NEAREST
            ), dtype=np.uint8
        ) >= 128

    return x[..., None], y


def metrics(y_true, y_pred):
    intersection = np.logical_and(
        y_true, y_pred
    ).sum()

    true_area = y_true.sum()
    pred_area = y_pred.sum()

    dice = (
        (2 * intersection + 1e-7)
        / (true_area + pred_area + 1e-7)
    )

    union = np.logical_or(
        y_true, y_pred
    ).sum()

    iou = (
        (intersection + 1e-7)
        / (union + 1e-7)
    )

    return float(dice), float(iou)


def collect_shenzhen():
    pairs = []

    for img in sorted(
        (SHEN / "images").glob("*.png")
    ):
        mask = SHEN / "masks" / img.name

        if not mask.exists():
            raise FileNotFoundError(mask)

        pairs.append((img, mask))

    if len(pairs) != 86:
        raise ValueError(
            f"Shenzhen test sayisi: {len(pairs)}"
        )

    return pairs


def collect_montgomery():
    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    names = [
        line.strip()
        for line in TEST_FILE.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    pairs = []

    for name in names:
        img = MONT / "images" / name
        mask = MONT / "masks" / name

        if not img.exists():
            raise FileNotFoundError(img)

        if not mask.exists():
            raise FileNotFoundError(mask)

        pairs.append((img, mask))

    if len(pairs) != 21:
        raise ValueError(
            f"Montgomery test sayisi: {len(pairs)}"
        )

    return pairs


datasets = {
    "Montgomery": collect_montgomery(),
    "Shenzhen": collect_shenzhen()
}

results = []

for model_name, model_path in MODELS.items():
    print(
        f"\nModel yukleniyor: {model_name}",
        flush=True
    )

    model = tf.keras.models.load_model(
        model_path, compile=False
    )

    for dataset_name, pairs in datasets.items():
        print(
            f"\n{model_name} - {dataset_name}: "
            f"{len(pairs)} goruntu",
            flush=True
        )

        scores = []

        for index, (img_path, mask_path) in enumerate(
            pairs, start=1
        ):
            x, y_true = load_pair(
                img_path, mask_path
            )

            prediction = model.predict(
                x[None, ...],
                verbose=0
            )[0, ..., 0]

            y_pred = prediction >= THRESHOLD

            dice, iou = metrics(
                y_true, y_pred
            )

            scores.append((dice, iou))

            results.append({
                "model": model_name,
                "dataset": dataset_name,
                "filename": img_path.name,
                "dice": round(dice, 6),
                "iou": round(iou, 6)
            })

            if index % 10 == 0:
                print(
                    f"{index}/{len(pairs)}",
                    flush=True
                )

        scores = np.asarray(scores)

        print(
            f"{model_name} | {dataset_name} | "
            f"Dice: {scores[:, 0].mean():.4f} | "
            f"IoU: {scores[:, 1].mean():.4f}",
            flush=True
        )

csv_path = OUT / "comparison_results.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model", "dataset",
            "filename", "dice", "iou"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nTum karsilastirmalar tamamlandi.")
print("CSV:", csv_path)
