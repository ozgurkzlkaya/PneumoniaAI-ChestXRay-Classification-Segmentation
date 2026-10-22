from pathlib import Path
import csv

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset" / "test"
OUTPUT = ROOT / "results" / "segmentation" / "mask_refinement_comparison"
MODEL_PATH = ROOT / "best_segmentation_model_v3.keras"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SEED = 42
SAMPLES_PER_CLASS = 5
THRESHOLD = 0.5

# Morfolojik islemler 256x256 maske uzerinde yapilir.
MIN_COMPONENT_PIXELS = 100
MAX_HOLE_PIXELS = 100
DILATION_ITERATIONS = 3


def select_images():
    rng = np.random.default_rng(SEED)
    selected = []

    for class_name in CLASSES:
        folder = DATASET / class_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        if len(files) < SAMPLES_PER_CLASS:
            raise RuntimeError(f"Yetersiz goruntu: {folder}")

        indices = rng.choice(
            len(files),
            size=SAMPLES_PER_CLASS,
            replace=False
        )

        selected.extend(
            (class_name, files[int(i)])
            for i in indices
        )

    return selected


def prepare_image(image):
    gray = image.convert("L").resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(gray, dtype=np.float32) / 255.0
    return array[np.newaxis, ..., np.newaxis]


def remove_small_components(mask, min_pixels):
    labels, count = ndi.label(mask)
    sizes = np.bincount(labels.ravel())

    result = mask.copy()

    for component_id in range(1, count + 1):
        if sizes[component_id] < min_pixels:
            result[labels == component_id] = False

    return result


def fill_small_holes(mask, max_pixels):
    inverse = ~mask
    labels, count = ndi.label(inverse)

    sizes = np.bincount(labels.ravel())

    border_labels = set(
        np.unique(
            np.concatenate([
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1]
            ])
        )
    )

    result = mask.copy()

    for component_id in range(1, count + 1):
        if component_id in border_labels:
            continue

        if sizes[component_id] <= max_pixels:
            result[labels == component_id] = True

    return result


def refine_mask(raw_mask):
    cleaned = remove_small_components(
        raw_mask,
        MIN_COMPONENT_PIXELS
    )

    cleaned = fill_small_holes(
        cleaned,
        MAX_HOLE_PIXELS
    )

    expanded = ndi.binary_dilation(
        cleaned,
        iterations=DILATION_ITERATIONS
    )

    return cleaned, expanded


def make_masked_image(image, mask):
    resized_mask = Image.fromarray(
        (mask.astype(np.uint8) * 255)
    ).resize(
        image.size,
        Image.Resampling.NEAREST
    )

    original_array = np.asarray(image, dtype=np.uint8)
    mask_array = np.asarray(resized_mask) > 0

    masked = original_array.copy()
    masked[~mask_array] = 0

    return Image.fromarray(masked)


def make_panel(image, masks, labels):
    tile_size = (300, 300)
    canvas = Image.new("RGB", (1200, 340), "white")
    draw = ImageDraw.Draw(canvas)

    images = [image]

    for mask in masks:
        images.append(make_masked_image(image, mask))

    for index, (tile, label) in enumerate(zip(images, labels)):
        resized = tile.resize(
            tile_size,
            Image.Resampling.BILINEAR
        )

        canvas.paste(resized, (index * 300, 40))
        draw.text(
            (index * 300 + 10, 12),
            label,
            fill="black"
        )

    return canvas


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    selected = select_images()

    print("U-Net V3 yukleniyor...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    if tuple(model.input_shape[1:]) != (256, 256, 1):
        raise ValueError(
            f"Beklenmeyen model girisi: {model.input_shape}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []

    for index, (class_name, path) in enumerate(
        selected,
        start=1
    ):
        with Image.open(path) as source:
            image = source.convert("RGB")

        prediction = model.predict(
            prepare_image(image),
            verbose=0
        )[0, :, :, 0]

        raw_mask = prediction >= THRESHOLD
        cleaned_mask, expanded_mask = refine_mask(raw_mask)

        masks = [
            raw_mask,
            cleaned_mask,
            expanded_mask
        ]

        labels = [
            "ORIGINAL",
            "A - RAW U-NET V3",
            "B - CLEANED",
            "C - EXPANDED"
        ]

        panel = make_panel(image, masks, labels)

        output_file = OUTPUT / f"{index:02d}_{class_name}.png"
        panel.save(output_file)

        raw_pct = float(raw_mask.mean() * 100)
        cleaned_pct = float(cleaned_mask.mean() * 100)
        expanded_pct = float(expanded_mask.mean() * 100)

        rows.append({
            "image": path.name,
            "class": class_name,
            "raw_coverage_pct": round(raw_pct, 2),
            "cleaned_coverage_pct": round(cleaned_pct, 2),
            "expanded_coverage_pct": round(expanded_pct, 2)
        })

        print(
            f"[{index:02d}/15] {class_name:10s} "
            f"Raw: %{raw_pct:.1f} | "
            f"Cleaned: %{cleaned_pct:.1f} | "
            f"Expanded: %{expanded_pct:.1f}"
        )

    csv_path = OUTPUT / "comparison_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nKarsilastirma tamamlandi.")
    print(f"Gorseller: {OUTPUT}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
