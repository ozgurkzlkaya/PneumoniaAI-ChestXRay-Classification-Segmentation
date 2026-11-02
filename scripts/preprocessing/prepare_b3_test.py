"""Prepare B3 alpha=.50 test images using the exact training mask pipeline.
Does not train or evaluate. Refuses existing output directory.
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from create_b3_softmask_datasets import mask_for, save_variant
    source = root / 'dataset' / 'test'
    out = root / 'results' / 'classification' / 'b3_test_alpha050_prepared'
    model_path = root / 'best_segmentation_model_v3.keras'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    entries = []
    for cls in CLASSES:
        folder = source / cls
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
        stems = [p.stem for p in files]
        if len(stems) != len(set(stems)):
            raise ValueError(f'PNG filename collisions in {folder}')
        print(f'{cls}: {len(files)}', flush=True)
        entries.extend((cls, p) for p in files)
    if len(entries) != 960:
        raise ValueError(f'Expected 960 test images, got {len(entries)}; no files written')
    print('Loading U-Net V3...', flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != (256, 256, 1):
        raise ValueError(f'Unexpected segmentation input: {model.input_shape}')
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for idx, (cls, src) in enumerate(entries, 1):
        with Image.open(src) as handle:
            image = handle.convert('RGB')
        mask = mask_for(image, model)
        target = out / cls / (src.stem + '.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        save_variant(image, mask, 0.50, target)
        rows.append({'class': cls, 'source': str(src), 'output': str(target),
                     'alpha': '0.50', 'mask_coverage_pct': round(float(mask.mean()*100), 4),
                     'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        if idx % 50 == 0 or idx == len(entries):
            print(f'{idx}/{len(entries)}', flush=True)
    with (out / 'test_manifest.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Prepared {len(rows)} test images: {out}', flush=True)
    print('No training. No evaluation. Existing files untouched.', flush=True)


if __name__ == '__main__':
    main()
