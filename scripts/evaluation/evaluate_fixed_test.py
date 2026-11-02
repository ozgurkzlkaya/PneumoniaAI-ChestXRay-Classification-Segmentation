"""One-shot, pre-specified comparison on 960 test images.
Read-only input/model access; refuses to overwrite results.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def read_image(path):
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    return tf.image.resize(image, (224, 224), method='bilinear')


def gather(root):
    keys, paths = [], []
    for i, cls in enumerate(CLASSES):
        folder = root / cls
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        items = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
        for p in items:
            keys.append((cls, p.stem))
            paths.append(p)
    if len(set(keys)) != len(keys):
        raise ValueError(f'Duplicate identities: {root}')
    return keys, paths


def predict(model_path, paths, batch_size):
    model = tf.keras.models.load_model(model_path, compile=False)
    probs = []
    for start in range(0, len(paths), batch_size):
        batch = tf.stack([read_image(p) for p in paths[start:start+batch_size]])
        scores = np.asarray(model(batch, training=False).numpy(), dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
            raise ValueError(f'Invalid output from {model_path}: {scores.shape}')
        if np.any(scores < -1e-5) or np.any(scores > 1.00001) or not np.allclose(scores.sum(axis=1), 1, atol=0.002):
            raise ValueError(f'Model outputs are not valid probabilities: {model_path}')
        probs.append(scores)
        if (start // batch_size + 1) % 5 == 0:
            print(f'  {min(start+batch_size, len(paths))}/{len(paths)}', flush=True)
    del model
    tf.keras.backend.clear_session()
    return np.concatenate(probs, axis=0)


def metrics(y, pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0,1,2], zero_division=0)
    return {'n': len(y), 'correct': int(np.sum(y == pred)),
            'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro')),
            'classes': {cls: {'precision': float(precision[i]), 'recall': float(recall[i]),
                              'f1': float(f1[i]), 'support': int(support[i])}
                        for i, cls in enumerate(CLASSES)},
            'confusion_matrix': confusion_matrix(y, pred, labels=[0,1,2]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error('batch-size must be >=1')
    root = args.root.resolve()
    exp = root / 'experiments' / 'classification_ablation'
    original = root / 'dataset' / 'test'
    masked = root / 'results' / 'classification' / 'b3_test_alpha050_prepared'
    out = root / 'results' / 'classification' / 'fixed_ensemble_test_evaluation'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing results: {out}')
    a_path = exp / 'original' / 'best_finetuned.keras'
    b_path = exp / 'b3_alpha050' / 'best_finetuned.keras'
    for p in (a_path, b_path, masked / 'test_manifest.csv'):
        if not p.is_file():
            raise FileNotFoundError(p)
    keys_a, paths_a = gather(original)
    keys_b, paths_b = gather(masked)
    if len(keys_a) != 960 or keys_a != keys_b:
        raise ValueError('Original and B3 test identities/order mismatch, or count != 960')
    with (masked / 'test_manifest.csv').open(newline='', encoding='utf-8-sig') as f:
        manifest = list(csv.DictReader(f))
    if len(manifest) != 960:
        raise ValueError('Test manifest does not contain 960 entries')
    expected = {(cls, p.stem): p for cls, p in zip((k[0] for k in keys_b), paths_b)}
    for row in manifest:
        key = (row['class'], Path(row['output']).stem)
        if key not in expected or expected[key].resolve() != Path(row['output']).resolve():
            raise ValueError(f'Manifest mismatch: {key}')
    y = np.array([CLASSES.index(k[0]) for k in keys_a], dtype=int)
    print('A original model...', flush=True)
    pa = predict(a_path, paths_a, args.batch_size)
    print('B3 alpha=.50 model...', flush=True)
    pb = predict(b_path, paths_b, args.batch_size)
    mean = 0.5 * pa + 0.5 * pb
    biased = mean.copy()
    biased[:, 1] += 0.01
    biased[:, 2] -= 0.01
    methods = {'A_original': pa, 'B3_alpha050': pb,
               'Ensemble_50_50': mean, 'Ensemble_50_50_bias_plus_002': biased}
    reports = {name: metrics(y, scores.argmax(axis=1)) for name, scores in methods.items()}
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'test_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'image_stem', 'original_path', 'b3_path',
                         *[f'a_prob_{c.lower()}' for c in CLASSES],
                         *[f'b3_prob_{c.lower()}' for c in CLASSES],
                         *[f'ensemble_prob_{c.lower()}' for c in CLASSES],
                         *[f'pred_{name}' for name in methods]])
        for i, (key, p_a, p_b) in enumerate(zip(keys_a, paths_a, paths_b)):
            writer.writerow([key[0], key[1], str(p_a), str(p_b),
                             *pa[i].tolist(), *pb[i].tolist(), *mean[i].tolist(),
                             *[CLASSES[int(scores[i].argmax())] for scores in methods.values()]])
    (out / 'metrics.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'n', 'correct', 'accuracy_pct', 'macro_f1',
                         'normal_recall_pct', 'bacterial_recall_pct', 'viral_recall_pct'])
        for name, result in reports.items():
            writer.writerow([name, result['n'], result['correct'],
                             round(100*result['accuracy'], 4), round(result['macro_f1'], 6),
                             *[round(100*result['classes'][c]['recall'], 4) for c in CLASSES]])
    for name, result in reports.items():
        cm = np.asarray(result['confusion_matrix'])
        with (out / f'{name}_confusion_matrix.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['true\\pred', *CLASSES])
            for cls, row in zip(CLASSES, cm):
                writer.writerow([cls, *row.tolist()])
        pred = methods[name].argmax(axis=1)
        report = classification_report(y, pred, labels=[0,1,2], target_names=CLASSES,
                                       digits=4, zero_division=0)
        (out / f'{name}_report.txt').write_text(report, encoding='utf-8')
        print(f'{name}: {result["correct"]}/960 accuracy={100*result["accuracy"]:.4f}% '
              f'macro_F1={result["macro_f1"]:.5f} '
              f'BACTERIAL_recall={100*result["classes"]["BACTERIAL"]["recall"]:.2f}% '
              f'VIRAL_recall={100*result["classes"]["VIRAL"]["recall"]:.2f}%', flush=True)
        print(cm, flush=True)
    print(f'Saved: {out}', flush=True)
    print('One-shot fixed test comparison. No training or parameter search.', flush=True)


if __name__ == '__main__':
    main()
