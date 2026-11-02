from pathlib import Path
import argparse
import csv
import json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import tensorflow as tf

SIZE = (256, 256)
THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)
MIN_AREAS = (0, 100, 250, 400)
HOLE_AREAS = (0, 150)


def load_pair(img_path, mask_path):
    with Image.open(img_path) as im:
        x = np.asarray(im.convert('L').resize(SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    with Image.open(mask_path) as im:
        y = np.asarray(im.convert('L').resize(SIZE, Image.Resampling.NEAREST), dtype=np.uint8) >= 128
    return x[..., None], y


def refine(mask, min_area, max_hole):
    if min_area == 0 and max_hole == 0:
        return mask
    structure = np.ones((3, 3), dtype=np.uint8)
    out = mask.copy()
    if min_area:
        labels, count = ndi.label(out, structure=structure)
        if count:
            areas = np.bincount(labels.ravel())
            keep = areas >= min_area
            keep[0] = False
            out = keep[labels]
    if max_hole:
        holes = ndi.binary_fill_holes(out) & ~out
        labels, count = ndi.label(holes, structure=structure)
        if count:
            areas = np.bincount(labels.ravel())
            fill = (areas > 0) & (areas <= max_hole)
            fill[0] = False
            out |= fill[labels]
    return out


def score(truth, pred):
    intersection = np.logical_and(truth, pred).sum()
    denom = truth.sum() + pred.sum()
    union = np.logical_or(truth, pred).sum()
    return (float(2 * intersection / denom) if denom else 1.0,
            float(intersection / union) if union else 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    mont = root / 'segmentation_dataset' / 'prepared'
    shen = root / 'segmentation_dataset' / 'prepared_shenzhen_v2'
    model_path = root / 'best_segmentation_model_v3.keras'
    test_file = root / 'segmentation_test_files.txt'
    out = root / 'results' / 'segmentation' / 'v3_validation_postprocessing'
    out.mkdir(parents=True, exist_ok=True)

    mont_images = sorted((mont / 'images').glob('*.png'))
    if len(mont_images) != 138:
        raise ValueError(f'Montgomery count mismatch: {len(mont_images)}')
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(mont_images))
    mont_train = [mont_images[i] for i in indices[:96]]
    mont_val = [mont_images[i] for i in indices[96:117]]
    mont_test = {mont_images[i].name for i in indices[117:]}
    saved_test = {line.strip() for line in test_file.read_text(encoding='utf-8-sig').splitlines() if line.strip()}
    if mont_test != saved_test:
        raise ValueError('Montgomery saved test split mismatch. Aborting.')
    if {p.name for p in mont_val} & saved_test or {p.name for p in mont_train} & saved_test:
        raise ValueError('Montgomery test overlap. Aborting.')

    shen_val = sorted((shen / 'val' / 'images').glob('*.png'))
    shen_test = {p.name for p in (shen / 'test' / 'images').glob('*.png')}
    shen_train = {p.name for p in (shen / 'train' / 'images').glob('*.png')}
    if len(shen_val) != 84 or len(shen_test) != 86 or len(shen_train) != 396:
        raise ValueError('Shenzhen split counts mismatch. Aborting.')
    if {p.name for p in shen_val} & (shen_test | shen_train):
        raise ValueError('Shenzhen filename overlap. Aborting.')

    pairs = []
    pairs += [('Montgomery', p, mont / 'masks' / p.name) for p in mont_val]
    pairs += [('Shenzhen', p, shen / 'val' / 'masks' / p.name) for p in shen_val]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    for dataset, image_path, mask_path in pairs:
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
    print('Loading U-Net V3 (no training)...', flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    samples = []
    for index, (dataset, image_path, mask_path) in enumerate(pairs, 1):
        x, truth = load_pair(image_path, mask_path)
        prob = np.asarray(model.predict(x[None, ...], verbose=0)[0, ..., 0], dtype=np.float32)
        samples.append((dataset, image_path.name, truth, prob))
        if index % 20 == 0 or index == len(pairs):
            print(f'Inference {index}/{len(pairs)}', flush=True)

    rows = []
    details = []
    for threshold in THRESHOLDS:
        for min_area in MIN_AREAS:
            for max_hole in HOLE_AREAS:
                grouped = {'Montgomery': [], 'Shenzhen': []}
                for dataset, name, truth, prob in samples:
                    pred = refine(prob >= threshold, min_area, max_hole)
                    dice, iou = score(truth, pred)
                    grouped[dataset].append((dice, iou))
                    details.append({'dataset': dataset, 'filename': name, 'threshold': threshold,
                                    'min_component_area': min_area, 'max_hole_area': max_hole,
                                    'dice': round(dice, 7), 'iou': round(iou, 7)})
                mont_scores = np.asarray(grouped['Montgomery'])
                shen_scores = np.asarray(grouped['Shenzhen'])
                mont_dice, mont_iou = mont_scores.mean(axis=0)
                shen_dice, shen_iou = shen_scores.mean(axis=0)
                rows.append({'threshold': threshold, 'min_component_area': min_area,
                             'max_hole_area': max_hole, 'montgomery_dice': round(float(mont_dice), 7),
                             'montgomery_iou': round(float(mont_iou), 7),
                             'shenzhen_dice': round(float(shen_dice), 7),
                             'shenzhen_iou': round(float(shen_iou), 7),
                             'balanced_mean_dice': round(float((mont_dice + shen_dice) / 2), 7),
                             'balanced_mean_iou': round(float((mont_iou + shen_iou) / 2), 7)})

    baseline = next(row for row in rows if row['threshold'] == 0.5 and row['min_component_area'] == 0 and row['max_hole_area'] == 0)
    best = max(rows, key=lambda r: (r['balanced_mean_dice'], -abs(r['threshold'] - 0.5), -r['min_component_area'], -r['max_hole_area']))
    def write_csv(path, items):
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    write_csv(out / 'validation_grid.csv', rows)
    write_csv(out / 'validation_per_image.csv', details)
    summary = {'validation_images': {'Montgomery': len(mont_val), 'Shenzhen': len(shen_val)},
               'selection_metric': 'equal-weight average of per-dataset mean Dice',
               'baseline': baseline, 'best_validation': best,
               'note': 'Parameters selected on validation only. Test images were not evaluated.'}
    (out / 'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nBASELINE (validation):', json.dumps(baseline, indent=2))
    print('\nBEST (validation):', json.dumps(best, indent=2))
    print('\nSaved:', out)
    print('No test evaluation performed. Original models and prior results unchanged.')

if __name__ == '__main__':
    main()
