from pathlib import Path
import argparse, csv, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import tensorflow as tf

THRESHOLD = 0.45
MIN_AREA = 400
MAX_HOLE = 150
SIZE = (256, 256)

def load_pair(image_path, mask_path):
    with Image.open(image_path) as im:
        x = np.asarray(im.convert('L').resize(SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    with Image.open(mask_path) as im:
        y = np.asarray(im.convert('L').resize(SIZE, Image.Resampling.NEAREST), dtype=np.uint8) >= 128
    return x[..., None], y

def refine(mask):
    structure = np.ones((3,3), dtype=np.uint8)
    labels, n = ndi.label(mask, structure=structure)
    if n:
        areas = np.bincount(labels.ravel())
        keep = areas >= MIN_AREA
        keep[0] = False
        mask = keep[labels]
    holes = ndi.binary_fill_holes(mask) & ~mask
    labels, n = ndi.label(holes, structure=structure)
    if n:
        areas = np.bincount(labels.ravel())
        fill = (areas > 0) & (areas <= MAX_HOLE)
        fill[0] = False
        mask |= fill[labels]
    return mask

def score(y, p):
    inter = np.logical_and(y, p).sum()
    denom = y.sum() + p.sum()
    union = np.logical_or(y, p).sum()
    return float(2*inter/denom) if denom else 1., float(inter/union) if union else 1.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.root.resolve()
    mont = root/'segmentation_dataset'/'prepared'
    shen = root/'segmentation_dataset'/'prepared_shenzhen_v2'
    images = sorted((mont/'images').glob('*.png'))
    if len(images) != 138: raise ValueError('Montgomery image count is not 138')
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(images))
    expected = {images[i].name for i in indices[117:]}
    saved = {x.strip() for x in (root/'segmentation_test_files.txt').read_text(encoding='utf-8-sig').splitlines() if x.strip()}
    if expected != saved: raise ValueError('Montgomery test split mismatch')
    mont_test = [images[i] for i in indices[117:]]
    shen_test = sorted((shen/'test'/'images').glob('*.png'))
    if len(shen_test) != 86: raise ValueError('Shenzhen test count is not 86')
    manifest = shen/'split_manifest.csv'
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    expected_shen = {r['filename'] for r in rows if r['split'] == 'test'}
    if {p.name for p in shen_test} != expected_shen: raise ValueError('Shenzhen manifest mismatch')
    pairs = [('Montgomery', p, mont/'masks'/p.name) for p in mont_test]
    pairs += [('Shenzhen', p, shen/'test'/'masks'/p.name) for p in shen_test]
    for ds, p, m in pairs:
        if not m.is_file(): raise FileNotFoundError(m)
    model_path = root/'best_segmentation_model_v3.keras'
    model = tf.keras.models.load_model(model_path, compile=False)
    results = []
    for i, (ds, p, m) in enumerate(pairs, 1):
        x, truth = load_pair(p, m)
        prob = np.asarray(model.predict(x[None, ...], verbose=0)[0,...,0], dtype=np.float32)
        baseline = prob >= .5
        optimized = refine(prob >= THRESHOLD)
        bd, bi = score(truth, baseline)
        od, oi = score(truth, optimized)
        results.append(dict(dataset=ds, filename=p.name, baseline_dice=bd, optimized_dice=od, delta_dice=od-bd, baseline_iou=bi, optimized_iou=oi, delta_iou=oi-bi))
        if i % 20 == 0 or i == len(pairs): print(f'Evaluated {i}/{len(pairs)}', flush=True)
    summary = {'configuration': {'baseline': {'threshold': .5, 'min_component_area': 0, 'max_hole_area': 0}, 'optimized': {'threshold': THRESHOLD, 'min_component_area': MIN_AREA, 'max_hole_area': MAX_HOLE}}, 'datasets': {}}
    for ds in ('Montgomery','Shenzhen'):
        subset = [r for r in results if r['dataset'] == ds]
        summary['datasets'][ds] = {'n': len(subset)}
        for key in ('baseline_dice','optimized_dice','delta_dice','baseline_iou','optimized_iou','delta_iou'):
            summary['datasets'][ds][key] = float(np.mean([r[key] for r in subset]))
    summary['balanced_mean_delta_dice'] = float(np.mean([summary['datasets'][ds]['delta_dice'] for ds in ('Montgomery','Shenzhen')]))
    summary['note'] = 'Parameters fixed on validation. Test sets have previously been used for V1/V2/V3 comparison; results are not untouched independent model-selection test.'
    out = root/'results'/'segmentation'/'v3_fixed_postprocessing_test'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'per_image.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nTEST SUMMARY (fixed parameters):\n'+json.dumps(summary, indent=2))
    print('\nSaved:', out)

if __name__ == '__main__': main()
