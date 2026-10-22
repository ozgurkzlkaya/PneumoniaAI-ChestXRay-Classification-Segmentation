
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

BASE = Path(__file__).resolve().parents[2]

SOURCE = BASE / "reference_segmentation" / "images"
OUTPUT = BASE / "results" / "segmentation" / "pneumonia_segmentation_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "V1": BASE / "best_segmentation_model.keras",
    "V2": BASE / "best_segmentation_model_v2.keras",
    "V3": BASE / "best_segmentation_model_v3.keras"
}

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
EXTENSIONS = {".png", ".jpg", ".jpeg"}
SIZE = (256, 256)

models = {}

for name, path in MODEL_FILES.items():
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"{name} yukleniyor...", flush=True)

    models[name] = tf.keras.models.load_model(
        path, compile=False
    )


def load_image(path):
    with Image.open(path) as im:
        image = im.convert("L").resize(
            SIZE, Image.Resampling.BILINEAR
        )

    return np.asarray(
        image, dtype=np.float32
    ) / 255.0


def make_overlay(image, mask):
    rgb = np.stack([image] * 3, axis=-1)
    result = rgb.copy()

    # Akciger maskesini yesil goster
    green = np.array([0.0, 1.0, 0.0])
    result[mask] = (
        0.65 * result[mask] + 0.35 * green
    )

    return np.clip(result, 0, 1)


for class_name in CLASSES:
    class_dir = SOURCE / class_name

    if not class_dir.exists():
        raise FileNotFoundError(class_dir)

    files = sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in EXTENSIONS
    )

    if not files:
        raise RuntimeError(
            f"Goruntu bulunamadi: {class_dir}"
        )

    class_output = OUTPUT / class_name
    class_output.mkdir(exist_ok=True)

    print(
        f"\n{class_name}: {len(files)} goruntu",
        flush=True
    )

    for index, path in enumerate(files, start=1):
        image = load_image(path)
        x = image[None, ..., None]

        fig, axes = plt.subplots(
            1, 4, figsize=(16, 4.5)
        )

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Orijinal")

        for col, (name, model) in enumerate(
            models.items(), start=1
        ):
            prediction = model.predict(
                x, verbose=0
            )[0, ..., 0]

            mask = prediction >= 0.5
            overlay = make_overlay(image, mask)

            axes[col].imshow(overlay)
            axes[col].set_title(
                f"U-Net {name}"
            )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{class_name} | {path.name}",
            fontsize=11
        )

        fig.tight_layout()

        output_path = (
            class_output
            / f"{index:02d}_{path.stem}.png"
        )

        fig.savefig(
            output_path,
            dpi=130,
            bbox_inches="tight"
        )
        plt.close(fig)

        print(
            f"{index}/{len(files)} tamamlandi",
            flush=True
        )

print("\nKARSILASTIRMA TAMAMLANDI")
print("Sonuclar:", OUTPUT)
