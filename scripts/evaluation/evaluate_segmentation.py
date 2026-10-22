
from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "segmentation_dataset" / "prepared"
MODEL = BASE / "best_segmentation_model.keras"
TEST_LIST = BASE / "segmentation_test_files.txt"
OUTPUT = BASE / "results" / "segmentation" / "segmentation_results"

OUTPUT.mkdir(exist_ok=True)

SIZE = 256
THRESHOLD = 0.5

# Kaydedilmis en iyi modeli yukle
model = tf.keras.models.load_model(
    MODEL,
    compile=False
)

test_names = [
    name.strip()
    for name in TEST_LIST.read_text(
        encoding="utf-8"
    ).splitlines()
    if name.strip()
]

assert len(test_names) == 21, (
    f"21 test goruntusu bekleniyordu: {len(test_names)}"
)


def load_image(path, is_mask=False):
    with Image.open(path) as image:
        image = image.convert("L")
        method = (
            Image.Resampling.NEAREST
            if is_mask
            else Image.Resampling.BILINEAR
        )
        image = image.resize((SIZE, SIZE), method)
        arr = np.asarray(image, dtype=np.float32)

    if is_mask:
        return arr > 127

    return arr / 255.0


def calculate_metrics(y_true, y_pred):
    intersection = np.logical_and(
        y_true, y_pred
    ).sum()

    true_area = y_true.sum()
    pred_area = y_pred.sum()

    union = np.logical_or(
        y_true, y_pred
    ).sum()

    dice = (
        (2 * intersection) / (true_area + pred_area)
        if true_area + pred_area > 0
        else 1.0
    )

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    return float(dice), float(iou)


def save_comparison(name, x, truth, pred, dice, iou):
    original = Image.fromarray(
        (x * 255).astype(np.uint8)
    ).convert("RGB")

    gt = Image.fromarray(
        truth.astype(np.uint8) * 255
    ).convert("RGB")

    prediction = Image.fromarray(
        pred.astype(np.uint8) * 255
    ).convert("RGB")

    # Yan yana uc goruntu
    canvas = Image.new(
        "RGB", (SIZE * 3, SIZE + 55), "white"
    )

    canvas.paste(original, (0, 35))
    canvas.paste(gt, (SIZE, 35))
    canvas.paste(prediction, (SIZE * 2, 35))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Original X-Ray", fill="black")
    draw.text((SIZE + 10, 10), "Ground Truth", fill="black")
    draw.text((SIZE * 2 + 10, 10), "U-Net Prediction", fill="black")

    draw.text(
        (10, SIZE + 38),
        f"Dice: {dice:.4f} | IoU: {iou:.4f}",
        fill="black"
    )

    canvas.save(OUTPUT / f"comparison_{name}")


results = []

print("Test basliyor...")

for index, name in enumerate(test_names, start=1):
    image_path = DATA / "images" / name
    mask_path = DATA / "masks" / name

    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError(name)

    x = load_image(image_path)
    truth = load_image(mask_path, is_mask=True)

    prediction = model.predict(
        x[None, ..., None],
        verbose=0
    )[0, ..., 0]

    pred_mask = prediction >= THRESHOLD

    dice, iou = calculate_metrics(
        truth, pred_mask
    )

    results.append({
        "filename": name,
        "dice": dice,
        "iou": iou
    })

    save_comparison(
        name, x, truth, pred_mask, dice, iou
    )

    print(
        f"[{index}/{len(test_names)}] {name} "
        f"Dice={dice:.4f} IoU={iou:.4f}",
        flush=True
    )

dice_values = np.array([
    r["dice"] for r in results
])

iou_values = np.array([
    r["iou"] for r in results
])

with open(
    OUTPUT / "segmentation_metrics.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=["filename", "dice", "iou"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 50)
print("BAGIMSIZ TEST SONUCLARI")
print("=" * 50)
print(f"Test goruntusu: {len(results)}")
print(f"Ortalama Dice: {dice_values.mean():.4f}")
print(f"Ortalama IoU : {iou_values.mean():.4f}")
print(f"Dice Std     : {dice_values.std():.4f}")
print(f"IoU Std      : {iou_values.std():.4f}")
print(f"Minimum Dice : {dice_values.min():.4f}")
print(f"Maksimum Dice: {dice_values.max():.4f}")
print(f"Sonuclar     : {OUTPUT}")
