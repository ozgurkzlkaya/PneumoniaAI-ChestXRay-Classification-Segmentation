"""Non-destructive validation diagnostic for original vs mask refinements.
Run from project root: python scripts/diagnostics/ablation_b_diagnostic.py --root . --limit-per-class 20
Uses validation only. Does not change any model or existing dataset.
"""
from pathlib import Path
import argparse, csv
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage as ndi

CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def segment(image, model):
    gray = image.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
    x = np.asarray(gray, dtype=np.float32)[None, ..., None] / 255.0
    pred = model.predict(x, verbose=0)[0, :, :, 0]
    return pred >= 0.5

def clean(mask):
    labels, n = ndi.label(mask)
    sizes = np.bincount(labels.ravel())
    result = mask.copy()
    for i in range(1, n + 1):
        if sizes[i] < 100:
            result[labels == i] = False
    inverse = ~result
    labels, n = ndi.label(inverse)
    sizes = np.bincount(labels.ravel())
    borders = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    for i in range(1, n + 1):
        if i not in borders and sizes[i] <= 100:
            result[labels == i] = True
    return result

def masked_image(image, mask, background_scale):
    full_mask = Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
    m = np.asarray(full_mask) > 0
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    out = rgb.copy()
    out[~m] = np.rint(out[~m].astype(np.float32) * background_scale).astype(np.uint8)
    return Image.fromarray(out)

def predict(model, image):
    # Matches tf.keras.utils.image_dataset_from_directory default RGB + bilinear 224x224 + 0..255 input.
    import io
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    x = tf.io.decode_png(buf.getvalue(), channels=3)
    x = tf.image.resize(x, (224, 224), method='bilinear')
    scores = np.asarray(model.predict(x[None, ...], verbose=0)[0], dtype=np.float64)
    return int(np.argmax(scores)), scores

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--limit-per-class', type=int, default=20, help='0 = entire validation set')
    args = p.parse_args()
    root = args.root.resolve()
    if args.limit_per_class < 0:
        p.error('--limit-per-class cannot be negative')
    val = root / 'dataset' / 'validation'
    seg_path = root / 'best_segmentation_model_v3.keras'
    b_path = root / 'experiments' / 'classification_ablation' / 'masked' / 'best_finetuned.keras'
    for path in (val, seg_path, b_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output = root / 'results' / 'classification' / 'ablation_b_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    print('Loading segmentation and masked classifier...')
    seg_model = tf.keras.models.load_model(seg_path, compile=False)
    classifier = tf.keras.models.load_model(b_path, compile=False)
    variants = ['raw', 'cleaned', 'expanded', 'context25']
    records = []
    for true_idx, cls in enumerate(CLASSES):
        files = sorted(f for f in (val / cls).iterdir() if f.suffix.lower() in EXTENSIONS)
        if args.limit_per_class:
            files = files[:args.limit_per_class]
        print(f'{cls}: {len(files)} images', flush=True)
        for k, path in enumerate(files, 1):
            with Image.open(path) as src:
                image = src.convert('RGB')
            raw = segment(image, seg_model)
            cleaned = clean(raw)
            expanded = ndi.binary_dilation(cleaned, iterations=3)
            masks = {'raw': raw, 'cleaned': cleaned, 'expanded': expanded, 'context25': expanded}
            for variant in variants:
                scale = 0.25 if variant == 'context25' else 0.0
                im = masked_image(image, masks[variant], scale)
                pred, scores = predict(classifier, im)
                records.append({'class': cls, 'file': path.name, 'variant': variant,
                                'true_idx': true_idx, 'pred_idx': pred, 'correct': int(pred == true_idx),
                                'mask_coverage_pct': round(float(masks[variant].mean() * 100), 3),
                                **{f'score_{c}': round(float(scores[i]), 6) for i, c in enumerate(CLASSES)}})
            if k % 10 == 0:
                print(f'  {k}/{len(files)}', flush=True)
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)
    summary = []
    for variant in variants:
        subset = [r for r in records if r['variant'] == variant]
        accuracy = np.mean([r['correct'] for r in subset])
        recalls = []
        f1s = []
        for idx in range(3):
            tp = sum(r['true_idx'] == idx and r['pred_idx'] == idx for r in subset)
            fp = sum(r['true_idx'] != idx and r['pred_idx'] == idx for r in subset)
            fn = sum(r['true_idx'] == idx and r['pred_idx'] != idx for r in subset)
            recalls.append(tp / (tp + fn) if tp + fn else 0)
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
        summary.append({'variant': variant, 'n': len(subset), 'accuracy_pct': round(float(accuracy * 100), 3),
                        'macro_f1': round(float(np.mean(f1s)), 5),
                        'recall_normal_pct': round(recalls[0] * 100, 2),
                        'recall_bacterial_pct': round(recalls[1] * 100, 2),
                        'recall_viral_pct': round(recalls[2] * 100, 2),
                        'mean_coverage_pct': round(float(np.mean([r['mask_coverage_pct'] for r in subset])), 2)})
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print('\nVALIDATION diagnostic (same B model, no retraining):')
    for row in summary:
        print(row)
    print(f'\nResults: {output}')
    print('CAUTION: context25 is out-of-distribution for the B model; exploratory only.')
    print('Do not use these comparisons to claim an improved trained model.')

if __name__ == '__main__':
    main()
