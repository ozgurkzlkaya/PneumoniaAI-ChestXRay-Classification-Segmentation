
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

BASE = Path(__file__).resolve().parents[2]

ROOT = (
    BASE / "segmentation_dataset"
    / "Shenzhen" / "Shenzhen"
)

IMG_DIR = ROOT / "img"
MASK_DIR = ROOT / "mask"

OUTPUT = (
    BASE / "results" / "segmentation" / "segmentation_results"
    / "shenzhen_mask_check.png"
)

OUTPUT.parent.mkdir(
    parents=True, exist_ok=True
)

files = sorted(IMG_DIR.glob("*.png"))

random.seed(42)
selected = random.sample(files, min(6, len(files)))

fig, axes = plt.subplots(
    len(selected), 3,
    figsize=(12, 4 * len(selected))
)

for i, image_path in enumerate(selected):
    mask_path = MASK_DIR / image_path.name

    with Image.open(image_path) as im:
        image = np.array(im.convert("L"))

    with Image.open(mask_path) as m:
        mask = np.array(m.convert("L"))

    # Sadece onizleme icin gecici esik.
    # Egitim esigi olarak henuz belirlenmedi.
    binary_mask = mask >= 128

    overlay = np.stack(
        [image] * 3, axis=-1
    ).astype(np.float32)

    overlay[binary_mask] = (
        0.55 * overlay[binary_mask]
        + 0.45 * np.array([0, 255, 0])
    )

    axes[i, 0].imshow(image, cmap="gray")
    axes[i, 0].set_title(image_path.name)

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title(
        f"Original Mask | "
        f"Min={mask.min()}, Max={mask.max()}"
    )

    axes[i, 2].imshow(
        np.clip(overlay, 0, 255).astype(np.uint8)
    )
    axes[i, 2].set_title(
        "Overlay (threshold=128)"
    )

    for ax in axes[i]:
        ax.axis("off")

    print(
        image_path.name,
        "| mask min:", mask.min(),
        "| max:", mask.max(),
        "| >=128:",
        round(binary_mask.mean() * 100, 2),
        "%",
        flush=True
    )

plt.tight_layout()
plt.savefig(
    OUTPUT, dpi=140,
    bbox_inches="tight"
)

plt.show()

print("\nGorsel kaydedildi:", OUTPUT)
