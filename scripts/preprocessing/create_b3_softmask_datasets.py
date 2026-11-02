"""Create isolated B3 alpha=0.25/0.50 train+validation datasets.
Uses exactly the B2 segmentation/refinement pipeline. Never overwrites existing datasets.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import numpy as np
import tensorflow as tf
from PIL import Image
from create_b2_expanded_dataset import CLASSES, SPLITS, EXTENSIONS, refine

VARIANTS = {'b3_alpha025': 0.25, 'b3_alpha050': 0.50}


def mask_for(image, model):
    gray = image.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
    x = np.asarray(gray, dtype=np.float32)[None, ..., None] / 255.0
    raw = model.predict(x, verbose=0)[0, :, :, 0] >= 0.5
    expanded = refine(raw)
    mask = Image.fromarray(expanded.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def save_variant(image, mask, alpha, destination):
    rgb = np.asarray(image, dtype=np.uint8).copy()
    rgb[~mask] = np.rint(rgb[~mask].astype(np.float32) * alpha).astype(np.uint8)
    Image.fromarray(rgb).save(destination, format='PNG')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--limit-per-class', type=int, default=0,
                        help='0=full dataset; positive=small smoke dataset in separate pilot folders')
    args = parser.parse_args()
    if args.limit_per_class < 0:
        parser.error('--limit-per-class must be nonnegative')
    root = args.root.resolve()
    source = root / 'dataset'
    model_path = root / 'best_segmentation_model_v3.keras'
    experiment = root / 'experiments' / 'classification_ablation'
    pilot = args.limit_per_class > 0
    dests = {name: experiment / name / ('pilot_dataset' if pilot else 'dataset') for name in VARIANTS}
    manifests = {name: experiment / name / ('pilot_manifest.csv' if pilot else 'manifest.csv') for name in VARIANTS}
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    for p in [*dests.values(), *manifests.values()]:
        if p.exists():
            raise FileExistsError(f'Refusing to overwrite existing output: {p}')
    entries = []
    for split in SPLITS:
        for cls in CLASSES:
            folder = source / split / cls
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if pilot:
                files = files[:args.limit_per_class]
            if not files:
                raise ValueError(f'No images in {folder}')
            stems = [p.stem for p in files]
            if len(stems) != len(set(stems)):
                raise ValueError(f'PNG filename collision in {folder}')
            entries.extend((split, cls, p) for p in files)
            print(f'{split}/{cls}: {len(files)}', flush=True)
    print(f'Processing {len(entries)} source images, {len(VARIANTS)} outputs each', flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != (256, 256, 1):
        raise ValueError(f'Unexpected U-Net input shape: {model.input_shape}')
    for d in dests.values():
        d.mkdir(parents=True, exist_ok=False)
    rows = {name: [] for name in VARIANTS}
    for index, (split, cls, path) in enumerate(entries, 1):
        with Image.open(path) as src:
            image = src.convert('RGB')
        mask = mask_for(image, model)
        coverage = round(float(mask.mean() * 100), 4)
        for name, alpha in VARIANTS.items():
            target = dests[name] / split / cls / (path.stem + '.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            save_variant(image, mask, alpha, target)
            rows[name].append({
                'split': split, 'class': cls,
                'source': str(path.relative_to(source)),
                'output': str(target.relative_to(dests[name])),
                'alpha': alpha, 'mask_coverage_pct': coverage,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            })
        if index % 100 == 0 or index == len(entries):
            print(f'{index}/{len(entries)}', flush=True)
    for name in VARIANTS:
        with manifests[name].open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[name][0]))
            writer.writeheader()
            writer.writerows(rows[name])
        print(f'{name}: {len(rows[name])} images; manifest: {manifests[name]}', flush=True)
    print('DONE: A/B/B2 models and datasets untouched. No training or test-set processing.')


if __name__ == '__main__':
    main()
