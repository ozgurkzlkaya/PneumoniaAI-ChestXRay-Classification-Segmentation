"""Validation-only, read-only model inference for conservative chest X-ray TTA.
No training, no test access, and no overwriting previous output.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def gather(folder):
    identities, paths = [], []
    for cls in CLASSES:
        sub = folder / cls
        if not sub.is_dir():
            raise FileNotFoundError(f'Missing class folder: {sub}')
        for p in sorted(sub.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTS:
                identities.append((cls, p.stem.lower()))
                paths.append(p)
    if len(set(identities)) != len(identities):
        raise ValueError(f'Duplicate image identities in {folder}')
    return identities, paths


def read_image(path):
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    return tf.image.resize(image, (224, 224), method='bilinear')


def transform(batch, variant):
    if variant == 'identity':
        return batch
    if variant == 'brightness_low':
        return tf.clip_by_value(batch * 0.98, 0.0, 255.0)
    if variant == 'brightness_high':
        return tf.clip_by_value(batch * 1.02, 0.0, 255.0)
    if variant in ('contrast_low', 'contrast_high'):
        factor = 0.98 if variant == 'contrast_low' else 1.02
        mean = tf.reduce_mean(batch, axis=[1, 2], keepdims=True)
        return tf.clip_by_value((batch - mean) * factor + mean, 0.0, 255.0)
    raise ValueError(variant)


def check_probs(scores, label):
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError(f'Invalid prediction shape/values: {label}')
    if np.any(scores < -1e-5) or np.any(scores > 1.00001) or not np.allclose(scores.sum(axis=1), 1, atol=0.002):
        raise ValueError(f'Outputs are not class probabilities: {label}')


def infer(model_path, paths, batch_size):
    print(f'Loading {model_path}', flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    variants = ('identity', 'brightness_low', 'brightness_high', 'contrast_low', 'contrast_high')
    outputs = {v: [] for v in variants}
    for start in range(0, len(paths), batch_size):
        batch = tf.stack([read_image(p) for p in paths[start:start + batch_size]])
        for variant in variants:
            scores = np.asarray(model(transform(batch, variant), training=False).numpy(), dtype=np.float64)
            check_probs(scores, f'{model_path} {variant}')
            outputs[variant].append(scores)
        if (start // batch_size + 1) % 5 == 0:
            print(f'  {min(start + batch_size, len(paths))}/{len(paths)}', flush=True)
    del model
    tf.keras.backend.clear_session()
    return {k: np.concatenate(v, axis=0) for k, v in outputs.items()}


def load_reference(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    mapping = {}
    with csv_path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            p = Path(row['path'])
            key = (row['true_class'].upper(), p.stem.lower())
            if key in mapping:
                raise ValueError(f'Duplicate CSV identity: {key}')
            mapping[key] = np.array([float(row['prob_normal']), float(row['prob_bacterial']), float(row['prob_viral'])])
    return mapping


def validate_reference(keys, new_probs, csv_path, tolerance):
    reference = load_reference(csv_path)
    if set(keys) != set(reference):
        raise ValueError(f'CSV/image identity mismatch for {csv_path}: missing={len(set(keys)-set(reference))}, extra={len(set(reference)-set(keys))}')
    expected = np.stack([reference[k] for k in keys])
    difference = float(np.max(np.abs(new_probs - expected)))
    changed = int(np.sum(np.argmax(new_probs, axis=1) != np.argmax(expected, axis=1)))
    print(f'Reference check: {csv_path.name} max_probability_difference={difference:.8f}, changed_predictions={changed}', flush=True)
    if difference > tolerance or changed:
        raise ValueError('Fresh identity predictions do not match stored validation probabilities. STOP: inspect preprocessing/model versions before interpreting TTA.')


def metrics(y, scores):
    pred = np.argmax(scores, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {
        'n': len(y), 'correct': int(np.sum(y == pred)),
        'accuracy': float(np.mean(y == pred)),
        'macro_f1': float(f1_score(y, pred, average='macro')),
        'classes': {cls: {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i, cls in enumerate(CLASSES)},
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Validation-only conservative TTA for A+B3 ensemble')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--reference-tolerance', type=float, default=0.001)
    args = parser.parse_args()
    if args.batch_size < 1 or args.reference_tolerance <= 0:
        parser.error('batch-size and reference-tolerance must be positive')
    root = args.root.resolve()
    exp = root / 'experiments' / 'classification_ablation'
    a_model = exp / 'original' / 'best_finetuned.keras'
    b_model = exp / 'b3_alpha050' / 'best_finetuned.keras'
    a_folder = root / 'dataset' / 'validation'
    b_folder = exp / 'b3_alpha050' / 'dataset' / 'validation'
    refs = root / 'results' / 'classification' / 'ablation_uniform_evaluation'
    out = root / 'results' / 'classification' / 'tta_validation_v1'
    if out.exists():
        raise FileExistsError(f'Output already exists; refusing to overwrite: {out}')
    for p in (a_model, b_model):
        if not p.is_file():
            raise FileNotFoundError(p)
    a_keys, a_paths = gather(a_folder)
    b_keys, b_paths = gather(b_folder)
    if len(a_keys) != 875 or a_keys != b_keys:
        raise ValueError(f'Validation identity/order mismatch: A={len(a_keys)} B3={len(b_keys)}; expected 875')
    y = np.array([CLASSES.index(cls) for cls, _ in a_keys], dtype=int)
    print('Class counts:', {cls: int(np.sum(y == i)) for i, cls in enumerate(CLASSES)}, flush=True)
    pa = infer(a_model, a_paths, args.batch_size)
    validate_reference(a_keys, pa['identity'], refs / 'A_original_predictions.csv', args.reference_tolerance)
    pb = infer(b_model, b_paths, args.batch_size)
    validate_reference(b_keys, pb['identity'], refs / 'B3_alpha050_predictions.csv', args.reference_tolerance)
    def ensemble(variants):
        a = np.mean(np.stack([pa[v] for v in variants]), axis=0)
        b = np.mean(np.stack([pb[v] for v in variants]), axis=0)
        return (a + b) / 2
    methods = {
        'baseline_identity': ensemble(('identity',)),
        'brightness_tta_3': ensemble(('identity', 'brightness_low', 'brightness_high')),
        'contrast_tta_3': ensemble(('identity', 'contrast_low', 'contrast_high')),
        'brightness_contrast_tta_5': ensemble(('identity', 'brightness_low', 'brightness_high', 'contrast_low', 'contrast_high')),
    }
    reports = {name: metrics(y, scores) for name, scores in methods.items()}
    if reports['baseline_identity']['correct'] != 703:
        raise ValueError(f'Expected 703/875 baseline correct; got {reports["baseline_identity"]["correct"]}. STOP.')
    print('\nVALIDATION ONLY: fixed TTA comparisons', flush=True)
    print(f'{"Method":31s} {"Correct":>7s} {"Accuracy":>10s} {"MacroF1":>9s} {"BactR":>9s} {"ViralR":>9s} {"Delta":>7s}')
    baseline_correct = reports['baseline_identity']['correct']
    for name, m in reports.items():
        print(f'{name:31s} {m["correct"]:7d} {100*m["accuracy"]:9.3f}% {m["macro_f1"]:9.5f} {100*m["classes"]["BACTERIAL"]["recall"]:8.2f}% {100*m["classes"]["VIRAL"]["recall"]:8.2f}% {m["correct"]-baseline_correct:+7d}', flush=True)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'metrics.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'n', 'correct', 'accuracy_pct', 'macro_f1', 'normal_recall_pct', 'bacterial_recall_pct', 'viral_recall_pct'])
        for name, m in reports.items():
            writer.writerow([name, m['n'], m['correct'], round(100*m['accuracy'], 4), round(m['macro_f1'], 6), *[round(100*m['classes'][cls]['recall'], 4) for cls in CLASSES]])
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'image_stem', *[f'{method}_pred' for method in methods]])
        for i, (cls, stem) in enumerate(a_keys):
            writer.writerow([cls, stem, *[CLASSES[int(scores[i].argmax())] for scores in methods.values()]])
    print(f'\nSaved: {out}', flush=True)
    print('No training. No test inference. No model changes.', flush=True)


if __name__ == '__main__':
    main()
