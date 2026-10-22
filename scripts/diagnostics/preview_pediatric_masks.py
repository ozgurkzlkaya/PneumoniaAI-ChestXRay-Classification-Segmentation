from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

ROOT = Path.cwd()
DATASET = ROOT / "dataset" / "test"
OUTPUT = ROOT / "results" / "segmentation" / "pediatric_mask_preview"
MODEL_PATH = ROOT / "best_segmentation_model_v3.keras"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42
SAMPLES_PER_CLASS = 5

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

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
        len(files), size=SAMPLES_PER_CLASS, replace=False
    )

    selected.extend(
        (class_name, files[int(i)]) for i in indices
    )

print("U-Net V3 yukleniyor...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Model giris/cikis boyutlarini dogrula.
if tuple(model.input_shape[1:]) != (256, 256, 1):
    raise ValueError(f"Beklenmeyen giris: {model.input_shape}")

OUTPUT.mkdir(parents=True, exist_ok=True)

for index, (class_name, path) in enumerate(selected, start=1):
    with Image.open(path) as source:
        image = source.convert("RGB")

    gray = image.convert("L").resize(
        (256, 256), Image.Resampling.BILINEAR
    )

    array = np.asarray(gray, dtype=np.float32) / 255.0
    batch = array[np.newaxis, ..., np.newaxis]

    prediction = model.predict(batch, verbose=0)[0, :, :, 0]

    mask = Image.fromarray(
        ((prediction >= 0.5).astype(np.uint8) * 255)
    ).resize(image.size, Image.Resampling.NEAREST)

    # Orijinal goruntunun dis bolgelerini siyaha cevir.
    original_array = np.asarray(image, dtype=np.uint8)
    mask_array = np.asarray(mask) > 0

    masked_array = original_array.copy()
    masked_array[~mask_array] = 0

    # Karsilastirma icin uc goruntuyu ayni boyuta getir.
    preview_size = (320, 320)

    original_preview = image.resize(preview_size)
    mask_preview = mask.resize(
        preview_size, Image.Resampling.NEAREST
    ).convert("RGB")
    masked_preview = Image.fromarray(masked_array).resize(
        preview_size
    )

    canvas = Image.new("RGB", (960, 355), "white")
    draw = ImageDraw.Draw(canvas)

    canvas.paste(original_preview, (0, 35))
    canvas.paste(mask_preview, (320, 35))
    canvas.paste(masked_preview, (640, 35))

    draw.text((10, 10), "ORIGINAL", fill="black")
    draw.text((330, 10), "U-NET V3 MASK", fill="black")
    draw.text((650, 10), "MASKED X-RAY", fill="black")

    output_path = OUTPUT / f"{index:02d}_{class_name}.png"
    canvas.save(output_path)

    mask_coverage = float(mask_array.mean() * 100)

    print(
        f"[{index:02d}/15] {class_name:10s} "
        f"mask coverage: %{mask_coverage:.1f}"
    )

print(f"\nTamamlandi. Sonuclar: {OUTPUT}")
