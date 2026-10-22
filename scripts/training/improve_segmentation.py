
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

BASE = Path(__file__).resolve().parents[2]

DATASET = BASE / "dataset" / "test"
MODEL_PATH = BASE / "best_segmentation_model.keras"
OUTPUT = BASE / "results" / "segmentation" / "segmentation_improved_results"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]

SIZE = 256
THRESHOLD = 0.5
SAMPLES_PER_CLASS = 5

# 256x256 maske icin baslangic parametreleri
MIN_COMPONENT_AREA = 400
MAX_HOLE_AREA = 150

OUTPUT.mkdir(exist_ok=True)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


def clean_mask(binary_mask):
    """
    Kucuk bagimsiz parcalari kaldirir.
    Kucuk kapali bosluklari doldurur.
    """
    mask = binary_mask.astype(bool)

    # 8-komsuluk ile bagli bilesen analizi
    structure = np.ones((3, 3), dtype=np.uint8)

    labels, count = ndi.label(
        mask, structure=structure
    )

    if count == 0:
        return mask

    areas = np.bincount(labels.ravel())
    areas[0] = 0

    # Belirli alandan kucuk parcalari kaldir
    keep = np.where(
        areas >= MIN_COMPONENT_AREA
    )[0]

    cleaned = np.isin(labels, keep)

    # Kucuk, kapali ic bosluklari doldur
    filled = ndi.binary_fill_holes(cleaned)
    holes = filled & ~cleaned

    hole_labels, hole_count = ndi.label(
        holes, structure=structure
    )

    if hole_count > 0:
        hole_areas = np.bincount(
            hole_labels.ravel()
        )
        hole_areas[0] = 0

        small_holes = np.where(
            (hole_areas > 0)
            & (hole_areas <= MAX_HOLE_AREA)
        )[0]

        cleaned |= np.isin(
            hole_labels, small_holes
        )

    return cleaned


def make_overlay(x, mask):
    rgb = np.stack([x, x, x], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    overlay = rgb.copy()

    green = np.array([0, 255, 0])

    overlay[mask] = (
        0.5 * overlay[mask] + 0.5 * green
    ).astype(np.uint8)

    return Image.fromarray(overlay)


def create_comparison(
    x, raw_mask, clean_mask_result, output_path
):
    original = Image.fromarray(
        (x * 255).astype(np.uint8)
    ).convert("RGB")

    raw = Image.fromarray(
        raw_mask.astype(np.uint8) * 255
    ).convert("RGB")

    cleaned = Image.fromarray(
        clean_mask_result.astype(np.uint8) * 255
    ).convert("RGB")

    raw_overlay = make_overlay(x, raw_mask)
    clean_overlay = make_overlay(
        x, clean_mask_result
    )

    images = [
        original,
        raw,
        cleaned,
        raw_overlay,
        clean_overlay
    ]

    titles = [
        "Original X-Ray",
        "Raw Mask",
        "Cleaned Mask",
        "Raw Overlay",
        "Cleaned Overlay"
    ]

    canvas = Image.new(
        "RGB",
        (SIZE * 5, SIZE + 40),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for i, (img, title) in enumerate(
        zip(images, titles)
    ):
        canvas.paste(img, (i * SIZE, 40))
        draw.text(
            (i * SIZE + 10, 12),
            title,
            fill="black"
        )

    canvas.save(output_path)


# Ayni 15 goruntuyu tekrar seciyoruz.
# Onceki kodda da seed=42 kullanilmisti.
rng = np.random.default_rng(42)

for class_name in CLASSES:
    folder = DATASET / class_name

    paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower()
        in [".jpg", ".jpeg", ".png"]
    )

    indices = rng.choice(
        len(paths),
        size=SAMPLES_PER_CLASS,
        replace=False
    )

    selected = [paths[i] for i in indices]

    class_output = OUTPUT / class_name
    class_output.mkdir(exist_ok=True)

    for index, path in enumerate(
        selected, start=1
    ):
        with Image.open(path) as image:
            image = image.convert("L")
            image = image.resize(
                (SIZE, SIZE),
                Image.Resampling.BILINEAR
            )

            x = np.asarray(
                image, dtype=np.float32
            ) / 255.0

        probability = model.predict(
            x[None, ..., None],
            verbose=0
        )[0, ..., 0]

        raw_mask = probability >= THRESHOLD

        improved_mask = clean_mask(raw_mask)

        removed_pixels = int(
            raw_mask.sum() - improved_mask.sum()
        )

        output_path = (
            class_output
            / f"{index:02d}_{path.stem}.png"
        )

        create_comparison(
            x,
            raw_mask,
            improved_mask,
            output_path
        )

        print(
            f"{class_name} {index}/5 | "
            f"{path.name} | "
            f"Raw={raw_mask.sum()} | "
            f"Cleaned={improved_mask.sum()} | "
            f"Pixel farki={removed_pixels}",
            flush=True
        )

print("\nMaske temizleme tamamlandi.")
print("Sonuclar:", OUTPUT)
