"""Validation-only B2 mask dilation sensitivity. Does not retrain or modify models."""
from pathlib import Path
import argparse
import csv
import numpy as np
import tensorflow as tf
from scipy import ndimage as ndi
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score
from ablation_b_diagnostic import CLASSES, EXTENSIONS, segment, clean, masked_image, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--limit-per-class', type=int, default=20, help='0 means full validation')
    args = parser.parse_args()
    if args.limit_per_class < 0:
        parser.error('--limit-per-class must be >= 0')
    root = args.root.resolve()
    val = root / 'dataset' / 'validation'
    seg_path = root / 'best_segmentation_model_v3.keras'
    b2_path = root / 'experiments' / 'classification_ablation' / 'b2_expanded' / 'best_finetuned.keras'
    for p in (val, seg_path, b2_path):
        if not p.exists():
            raise FileNotFoundError(p)
    output = root / 'results' / 'classification' / 'b2_mask_sensitivity'
    output.mkdir(parents=True, exist_ok=True)
    print('Loading U-Net and B2 classifier...', flush=True)
    seg_model = tf.keras.models.load_model(seg_path, compile=False)
    classifier = tf.keras.models.load_model(b2_path, compile=False)
    iterations = (0, 1, 3, 5, 8)
    records = []
    for true_idx, cls in enumerate(CLASSES):
        files = sorted(p for p in (val / cls).iterdir() if p.suffix.lower() in EXTENSIONS)
        if args.limit_per_class:
            files = files[:args.limit_per_class]
        print(f'{cls}: {len(files)} images', flush=True)
        for n, path in enumerate(files, 1):
            with Image.open(path) as src:
                image = src.convert('RGB')
            base = clean(segment(image, seg_model))
            for iteration in iterations:
                mask = base if iteration == 0 else ndi.binary_dilation(base, iterations=iteration)
                pred, scores = predict(classifier, masked_image(image, mask, 0.0))
                records.append({
                    'true_class': cls,
                    'filename': path.name,
                    'iterations': iteration,
                    'predicted_class': CLASSES[pred],
                    'correct': int(pred == true_idx),
                    'mask_coverage_pct': round(float(mask.mean()) * 100, 4),
                    **{f'prob_{c.lower()}': float(scores[i]) for i, c in enumerate(CLASSES)},
                })
            if n % 20 == 0:
                print(f'  {n}/{len(files)}', flush=True)
    if not records:
        raise RuntimeError('No validation images found')
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for iteration in iterations:
        rows = [r for r in records if r['iterations'] == iteration]
        yt = [CLASSES.index(r['true_class']) for r in rows]
        yp = [CLASSES.index(r['predicted_class']) for r in rows]
        recalls = recall_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        summary.append({
            'iterations': iteration,
            'n': len(rows),
            'accuracy_pct': round(100 * accuracy_score(yt, yp), 3),
            'macro_f1': round(f1_score(yt, yp, labels=[0, 1, 2], average='macro', zero_division=0), 5),
            'recall_normal_pct': round(100 * recalls[0], 3),
            'recall_bacterial_pct': round(100 * recalls[1], 3),
            'recall_viral_pct': round(100 * recalls[2], 3),
            'viral_prediction_pct': round(100 * sum(p == 2 for p in yp) / len(yp), 3),
            'normal_to_viral_count': sum(a == 0 and b == 2 for a, b in zip(yt, yp)),
            'mean_coverage_pct': round(float(np.mean([r['mask_coverage_pct'] for r in rows])), 3),
        })
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print('\nB2 MASK SENSITIVITY (validation, no retraining):')
    for row in summary:
        print(row)
    print('\nResults:', output)
    print('Note: 3 iterations should be compared against the B2 validation baseline; differences may indicate a preprocessing mismatch.')

if __name__ == '__main__':
    main()
