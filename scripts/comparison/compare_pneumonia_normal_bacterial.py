
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

BASE = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE / "reference_segmentation" / "images"
OUTPUT_DIR = BASE / "results" / "segmentation" / "segmentation_pneumonia_normal_bacterial"

MODELS = {
    "V1": BASE / "best_segmentation_model.keras",
    "V2": BASE / "best_segmentation_model_v2.keras",
    "V3": BASE / "best_segmentation_model_v3.keras",
}

CLASSES = ["NORMAL", "BACTERIAL"]
IMG_SIZE = (256, 256)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_image(path):
    with Image.open(path) as img:
        original = img.convert("RGB").copy()
        gray = img.convert("L").resize(
            IMG_SIZE, Image.Resampling.BILINEAR
        )
    x = np.asarray(gray, dtype=np.float32) / 255.0
    return original, x[None, ..., None]

def predict_mask(model, x):
    prediction = model.predict(x, verbose=0)[0, ..., 0]
    return prediction >= 0.5

def create_overlay(original, mask):
    rgb = np.asarray(original, dtype=np.uint8).copy()

    mask_img = Image.fromarray(
        (mask.astype(np.uint8) * 255)
    ).resize(original.size, Image.Resampling.NEAREST)

    binary = np.asarray(mask_img) >= 128
    result = rgb.astype(np.float32)

    green = np.array([0, 255, 0], dtype=np.float32)
    result[binary] = (
        0.55 * result[binary] + 0.45 * green
    )

    return np.clip(result, 0, 255).astype(np.uint8)

def main():
    loaded_models = {}

    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"Model bulunamadi: {path}")

        print(f"Model yukleniyor: {name}", flush=True)
        loaded_models[name] = tf.keras.models.load_model(
            path, compile=False
        )

    for class_name in CLASSES:
        class_dir = INPUT_DIR / class_name

        if not class_dir.exists():
            print(f"Klasor bulunamadi: {class_dir}")
            continue

        paths = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )

        output_class = OUTPUT_DIR / class_name
        output_class.mkdir(parents=True, exist_ok=True)

        print(
            f"\n{class_name}: {len(paths)} goruntu",
            flush=True
        )

        for index, path in enumerate(paths, 1):
            original, x = load_image(path)

            fig, axes = plt.subplots(
                1, 4, figsize=(20, 5)
            )

            axes[0].imshow(original)
            axes[0].set_title("Orijinal")

            for ax, (name, model) in zip(
                axes[1:], loaded_models.items()
            ):
                mask = predict_mask(model, x)
                overlay = create_overlay(original, mask)
                ax.imshow(overlay)
                ax.set_title(f"U-Net {name}")

            for ax in axes:
                ax.axis("off")

            fig.suptitle(
                f"{class_name} | {path.name}",
                fontsize=15
            )
            fig.tight_layout()

            output_path = output_class / (
                f"{index:02d}_{path.stem}.png"
            )

            fig.savefig(
                output_path,
                dpi=130,
                bbox_inches="tight"
            )
            plt.close(fig)

            print(
                f"{index}/{len(paths)} - {path.name}",
                flush=True
            )

    print("\nTum karsilastirmalar tamamlandi.")
    print(f"Sonuclar: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
