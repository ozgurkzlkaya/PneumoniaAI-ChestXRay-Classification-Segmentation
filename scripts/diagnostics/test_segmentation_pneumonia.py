
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[2]

DATASET = BASE / "dataset" / "test"
MODEL_PATH = BASE / "best_segmentation_model.keras"
OUTPUT = BASE / "results" / "segmentation" / "segmentation_pneumonia_results"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
SAMPLES_PER_CLASS = 5
SIZE = 256
THRESHOLD = 0.5

OUTPUT.mkdir(exist_ok=True)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

rng = np.random.default_rng(42)

for class_name in CLASSES:
    class_dir = DATASET / class_name

    image_paths = sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    if len(image_paths) < SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"{class_name}: Yeterli goruntu yok!"
        )

    indices = rng.choice(
        len(image_paths),
        size=SAMPLES_PER_CLASS,
        replace=False
    )

    selected = [image_paths[i] for i in indices]

    class_output = OUTPUT / class_name
    class_output.mkdir(exist_ok=True)

    for index, path in enumerate(selected, start=1):
        with Image.open(path) as image:
            image = image.convert("L")
            image = image.resize(
                (SIZE, SIZE),
                Image.Resampling.BILINEAR
            )
            x = np.asarray(
                image, dtype=np.float32
            ) / 255.0

        probabilities = model.predict(
            x[None, ..., None],
            verbose=0
        )[0, ..., 0]

        mask = probabilities >= THRESHOLD

        original = Image.fromarray(
            (x * 255).astype(np.uint8)
        ).convert("RGB")

        mask_image = Image.fromarray(
            mask.astype(np.uint8) * 255
        ).convert("RGB")

        overlay = np.asarray(original).copy()

        # Akciger bolgelerini yesil renkle isaretle
        overlay[mask] = (
            0.5 * overlay[mask]
            + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        overlay_image = Image.fromarray(overlay)

        canvas = Image.new(
            "RGB",
            (SIZE * 3, SIZE + 40),
            "white"
        )

        canvas.paste(original, (0, 40))
        canvas.paste(mask_image, (SIZE, 40))
        canvas.paste(overlay_image, (SIZE * 2, 40))

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, 12), "Original X-Ray", fill="black"
        )
        draw.text(
            (SIZE + 10, 12), "Predicted Mask", fill="black"
        )
        draw.text(
            (SIZE * 2 + 10, 12), "Overlay", fill="black"
        )

        output_name = f"{index:02d}_{path.stem}.png"
        canvas.save(class_output / output_name)

        print(
            f"{class_name}: {index}/{SAMPLES_PER_CLASS} "
            f"- {path.name}",
            flush=True
        )

print("\nTum siniflar icin segmentasyon tamamlandi.")
print("Sonuclar:", OUTPUT)
