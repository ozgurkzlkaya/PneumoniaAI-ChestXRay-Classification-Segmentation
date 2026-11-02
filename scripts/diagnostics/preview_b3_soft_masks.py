"""PneumoniaAI: non-destructive B3 soft-mask preview (no training or dataset writes).
Uses exactly the mask construction from create_b2_expanded_dataset.py.
"""
from pathlib import Path
import argparse
import csv
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from create_b2_expanded_dataset import refine, CLASSES, EXTENSIONS


def fit(image, size=(270, 270)):
    im = image.copy()
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, (24, 28, 36))
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def make_mask(image, model):
    gray = image.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
    x = np.asarray(gray, dtype=np.float32)[None, ..., None] / 255.0
    prediction = model.predict(x, verbose=0)[0, :, :, 0]
    expanded = refine(prediction >= 0.5)  # includes 3-iteration dilation
    resized = Image.fromarray((expanded.astype(np.uint8) * 255)).resize(image.size, Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def soft_mask(image, mask, alpha):
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    if alpha == 1:
        return Image.fromarray(rgb)
    rgb[~mask] = np.rint(rgb[~mask].astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='validation')
    parser.add_argument('--per-class', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error('--per-class must be >= 1')
    root = args.root.resolve()
    model_file = root / 'best_segmentation_model_v3.keras'
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    output = root / 'results' / 'segmentation' / 'b3_softmask_preview'
    output.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_file, compile=False)
    rng = random.Random(args.seed)
    records = []
    for cls in CLASSES:
        folder = root / 'dataset' / args.split / cls
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(files) < args.per_class:
            raise ValueError(f'{cls}: only {len(files)} images')
        selected = sorted(rng.sample(files, args.per_class))
        page = Image.new('RGB', (1140, 305 * len(selected) + 50), (246, 247, 250))
        draw = ImageDraw.Draw(page)
        draw.text((12, 12), f'{cls} | {args.split} | original / B2 alpha=0 / B3 alpha=0.25 / B3 alpha=0.50', fill=(0, 0, 0))
        for i, path in enumerate(selected):
            with Image.open(path) as src:
                original = src.convert('RGB')
            mask = make_mask(original, model)
            variants = [original, soft_mask(original, mask, 0), soft_mask(original, mask, .25), soft_mask(original, mask, .50)]
            y = 40 + 305 * i
            for j, im in enumerate(variants):
                page.paste(fit(im), (12 + j * 282, y))
            draw.text((12, y + 274), f'{i+1}. {path.name[:60]} | mask coverage {mask.mean()*100:.2f}%', fill=(0, 0, 0))
            records.append({'split': args.split, 'class': cls, 'filename': path.name, 'mask_coverage_pct': round(float(mask.mean()*100), 4)})
            print(f'{cls}: {i+1}/{args.per_class} {path.name}', flush=True)
        target = output / f'{args.split}_{cls.lower()}_b3_preview.png'
        page.save(target)
        print('Saved:', target, flush=True)
    csv_file = output / f'{args.split}_preview_manifest.csv'
    with csv_file.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('Manifest:', csv_file)
    print('Preview only: no model training, no modification to A/B/B2 datasets.')


if __name__ == '__main__':
    main()
