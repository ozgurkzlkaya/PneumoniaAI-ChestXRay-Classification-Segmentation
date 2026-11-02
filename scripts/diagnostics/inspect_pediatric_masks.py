"""Non-destructive pediatric lung-mask visual audit for PneumoniaAI.
Creates 30-image contact sheets and a CSV; does not change models/datasets.
"""
from pathlib import Path
import argparse
import csv
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

CLASSES = ("NORMAL", "BACTERIAL", "VIRAL")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def refine(raw):
    labels, n = ndi.label(raw)
    sizes = np.bincount(labels.ravel())
    cleaned = raw.copy()
    for idx in range(1, n + 1):
        if sizes[idx] < 100:
            cleaned[labels == idx] = False
    labels, n = ndi.label(~cleaned)
    sizes = np.bincount(labels.ravel())
    border = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    for idx in range(1, n + 1):
        if idx not in border and sizes[idx] <= 100:
            cleaned[labels == idx] = True
    return cleaned


def process(image, model):
    gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    x = np.asarray(gray, dtype=np.float32)[None, ..., None] / 255.0
    probability = model.predict(x, verbose=0)[0, :, :, 0]
    clean = refine(probability >= 0.5)
    expanded = ndi.binary_dilation(clean, iterations=3)
    raw_mask = Image.fromarray((clean.astype(np.uint8) * 255)).resize(image.size, Image.Resampling.NEAREST)
    expanded_mask = Image.fromarray((expanded.astype(np.uint8) * 255)).resize(image.size, Image.Resampling.NEAREST)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    raw_arr = np.asarray(raw_mask) > 0
    exp_arr = np.asarray(expanded_mask) > 0
    overlay = rgb.copy()
    overlay[raw_arr] = np.clip(0.55 * rgb[raw_arr].astype(np.float32) + 0.45 * np.array([0, 255, 90]), 0, 255).astype(np.uint8)
    # Mark outer 3-iteration dilation ring in amber, making boundary changes visible.
    ring = exp_arr & ~raw_arr
    overlay[ring] = np.clip(0.45 * rgb[ring].astype(np.float32) + 0.55 * np.array([255, 180, 0]), 0, 255).astype(np.uint8)
    masked = rgb.copy()
    masked[~exp_arr] = 0
    components = ndi.label(clean)[1]
    return Image.fromarray(overlay), Image.fromarray(masked), float(raw_arr.mean()*100), float(exp_arr.mean()*100), components


def fit(im, size=(300, 300)):
    im = im.copy()
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (25, 29, 37))
    canvas.paste(im, ((size[0]-im.width)//2, (size[1]-im.height)//2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", choices=["train", "validation"], default="validation")
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = args.root.resolve()
    model_file = root / "best_segmentation_model_v3.keras"
    data = root / "dataset" / args.split
    if args.per_class < 1:
        ap.error("--per-class must be positive")
    if not model_file.is_file() or not data.is_dir():
        raise FileNotFoundError(f"Missing model or dataset: {model_file} / {data}")
    output = root / "results" / "segmentation" / "pediatric_visual_audit"
    output.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_file, compile=False)
    rng = random.Random(args.seed)
    rows = []
    for cls in CLASSES:
        files = sorted(p for p in (data / cls).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(files) < args.per_class:
            raise ValueError(f"Not enough files in {cls}: {len(files)}")
        selected = sorted(rng.sample(files, args.per_class))
        page = Image.new("RGB", (940, 350 * len(selected) + 55), (248, 249, 252))
        draw = ImageDraw.Draw(page)
        draw.text((15, 12), f"{cls} — {args.split} | original / overlay / B2 masked", fill=(0, 0, 0))
        for idx, path in enumerate(selected):
            with Image.open(path) as src:
                image = src.convert("RGB")
            overlay, masked, raw_pct, exp_pct, components = process(image, model)
            y = 50 + idx * 350
            for j, im in enumerate((image, overlay, masked)):
                page.paste(fit(im), (15 + j*310, y))
            draw.text((15, y + 305), f"{idx+1}. {path.name[:48]}", fill=(0, 0, 0))
            draw.text((15, y + 322), f"mask {raw_pct:.1f}% -> B2 {exp_pct:.1f}% | regions={components}", fill=(35, 40, 50))
            rows.append({"split": args.split, "class": cls, "filename": path.name,
                         "original_size": f"{image.width}x{image.height}",
                         "mask_coverage_pct": round(raw_pct, 3),
                         "expanded_coverage_pct": round(exp_pct, 3),
                         "mask_components": components})
            print(f"{cls}: {idx+1}/{len(selected)} {path.name}", flush=True)
        target = output / f"{args.split}_{cls.lower()}_audit.png"
        page.save(target)
        print("Saved:", target, flush=True)
    csv_path = output / f"{args.split}_audit_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Metrics:", csv_path)
    print("NOTE: This is a visual audit, not ground-truth segmentation accuracy.")

if __name__ == "__main__":
    main()
