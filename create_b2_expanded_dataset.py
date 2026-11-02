"""Create an isolated B2 expanded-mask dataset; never overwrite existing images.
Usage: python create_b2_expanded_dataset.py --root "."
"""
from pathlib import Path
import argparse
import csv
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage as ndi

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
SPLITS = ('train', 'validation')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

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
    return ndi.binary_dilation(cleaned, iterations=3)

def process(image, model):
    gray = image.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
    x = np.asarray(gray, dtype=np.float32)[None, ..., None] / 255.0
    prediction = model.predict(x, verbose=0)[0, :, :, 0]
    expanded = refine(prediction >= 0.5)
    mask = Image.fromarray((expanded.astype(np.uint8) * 255)).resize(image.size, Image.Resampling.NEAREST)
    mask_arr = np.asarray(mask) > 0
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    rgb[~mask_arr] = 0
    return Image.fromarray(rgb), float(mask_arr.mean() * 100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--limit-per-class', type=int, default=0, help='0: all images; positive: pilot per split and class')
    args = parser.parse_args()
    if args.limit_per_class < 0:
        parser.error('--limit-per-class must be nonnegative')
    root = args.root.resolve()
    source = root / 'dataset'
    model_path = root / 'best_segmentation_model_v3.keras'
    base = root / 'experiments' / 'classification_ablation' / 'b2_expanded'
    destination = base / ('pilot_dataset' if args.limit_per_class else 'dataset')
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if destination.exists():
        raise FileExistsError(f'Destination already exists; refusing to overwrite: {destination}')
    entries = []
    for split in SPLITS:
        for cls in CLASSES:
            folder = source / split / cls
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if args.limit_per_class:
                files = files[:args.limit_per_class]
            if not files:
                raise RuntimeError(f'No images: {folder}')
            entries.extend((split, cls, p) for p in files)
    print(f'Generating {len(entries)} images at {destination}', flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != (256, 256, 1):
        raise ValueError(f'Unexpected segmentation input shape: {model.input_shape}')
    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for i, (split, cls, path) in enumerate(entries, 1):
        with Image.open(path) as source_image:
            image = source_image.convert('RGB')
        output, coverage = process(image, model)
        target = destination / split / cls / (path.stem + '.png')
        if target.exists():
            raise FileExistsError(f'Output filename collision: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        output.save(target)
        manifest.append({'split': split, 'class': cls, 'source': str(path.relative_to(source)),
                         'output': str(target.relative_to(destination)), 'coverage_pct': round(coverage, 3)})
        if i % 50 == 0 or i == len(entries):
            print(f'{i}/{len(entries)}', flush=True)
    with (base / ('pilot_manifest.csv' if args.limit_per_class else 'manifest.csv')).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print('Done. Original datasets and models untouched.')
    print(f'Dataset: {destination}')

if __name__ == '__main__':
    main()
