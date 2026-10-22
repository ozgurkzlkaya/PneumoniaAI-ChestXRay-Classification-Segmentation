
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage as ndi

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "segmentation_dataset" / "prepared"
TEST_FILE = BASE / "segmentation_test_files.txt"
MODEL_FILE = BASE / "best_segmentation_model.keras"

SIZE = 256
MIN_COMPONENT_AREA = 400
MAX_HOLE_AREA = 150

def clean_mask(mask):
    mask = mask.astype(bool)
    structure = np.ones((3, 3), dtype=np.uint8)

    labels, count = ndi.label(mask, structure)

    if count == 0:
        return mask

    areas = np.bincount(labels.ravel())
    areas[0] = 0

    keep = np.where(areas >= MIN_COMPONENT_AREA)[0]
    cleaned = np.isin(labels, keep)

    filled = ndi.binary_fill_holes(cleaned)
    holes = filled & ~cleaned

    hole_labels, count = ndi.label(holes, structure)

    if count > 0:
        hole_areas = np.bincount(hole_labels.ravel())
        hole_areas[0] = 0

        small_holes = np.where(
            (hole_areas > 0)
            & (hole_areas <= MAX_HOLE_AREA)
        )[0]

        cleaned |= np.isin(hole_labels, small_holes)

    return cleaned

def metrics(truth, prediction):
    intersection = np.logical_and(truth, prediction).sum()
    union = np.logical_or(truth, prediction).sum()
    total = truth.sum() + prediction.sum()

    dice = 2 * intersection / total if total else 1.0
    iou = intersection / union if union else 1.0

    return float(dice), float(iou)

model = tf.keras.models.load_model(
    MODEL_FILE, compile=False
)

names = [
    line.strip()
    for line in TEST_FILE.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

raw_results = []
clean_results = []

for i, name in enumerate(names, 1):
    with Image.open(DATA / "images" / name) as img:
        x = np.asarray(
            img.convert("L").resize(
                (SIZE, SIZE),
                Image.Resampling.BILINEAR
            ),
            dtype=np.float32
        ) / 255.0

    with Image.open(DATA / "masks" / name) as img:
        truth = np.asarray(
            img.convert("L").resize(
                (SIZE, SIZE),
                Image.Resampling.NEAREST
            )
        ) > 127

    probability = model.predict(
        x[None, ..., None], verbose=0
    )[0, ..., 0]

    raw = probability >= 0.5
    cleaned = clean_mask(raw)

    raw_score = metrics(truth, raw)
    clean_score = metrics(truth, cleaned)

    raw_results.append(raw_score)
    clean_results.append(clean_score)

    print(
        f"[{i}/{len(names)}] {name} | "
        f"Raw Dice={raw_score[0]:.4f} | "
        f"Clean Dice={clean_score[0]:.4f}",
        flush=True
    )

raw_mean = np.mean(raw_results, axis=0)
clean_mean = np.mean(clean_results, axis=0)

print("\n" + "=" * 55)
print("HAM VE TEMIZLENMIS SEGMENTASYON KARSILASTIRMASI")
print("=" * 55)
print(f"Test goruntusu       : {len(names)}")
print(f"Ham Dice             : {raw_mean[0]:.4f}")
print(f"Temizlenmis Dice     : {clean_mean[0]:.4f}")
print(f"Dice farki           : {clean_mean[0]-raw_mean[0]:+.4f}")
print(f"Ham IoU              : {raw_mean[1]:.4f}")
print(f"Temizlenmis IoU      : {clean_mean[1]:.4f}")
print(f"IoU farki            : {clean_mean[1]-raw_mean[1]:+.4f}")
