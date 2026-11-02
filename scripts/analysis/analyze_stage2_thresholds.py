"""Validation-only Stage 2 threshold sensitivity; no training, no test access."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from train_two_stage import collect, predict_prob, CLASSES


def adjust_viral_probability(q, threshold):
    # Odds correction: adjusted probability is 0.5 when original q == threshold.
    q = np.clip(q.astype(np.float64), 1e-7, 1 - 1e-7)
    odds = q / (1 - q)
    threshold_odds = threshold / (1 - threshold)
    return odds / (odds + threshold_odds)


def assess(p1, p2, y, threshold):
    viral = adjust_viral_probability(p2[:, 1], threshold)
    probs = np.column_stack([p1[:, 0], p1[:, 1] * (1 - viral), p1[:, 1] * viral])
    pred = np.argmax(probs, axis=1)
    report = classification_report(y, pred, labels=[0, 1, 2], target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    result = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro')),
        'normal_recall': float(report['NORMAL']['recall']),
        'bacterial_recall': float(report['BACTERIAL']['recall']),
        'viral_recall': float(report['VIRAL']['recall']),
        'bacterial_f1': float(report['BACTERIAL']['f1-score']),
        'viral_f1': float(report['VIRAL']['f1-score']),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--previous-run', required=True)
    parser.add_argument('--improvement-run', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    previous = (root / args.previous_run).resolve()
    improvement = (root / args.improvement_run).resolve()
    stage1_path = previous / 'stage1_selected.keras'
    candidates = {
        'baseline': previous / 'stage2_selected.keras',
        'low_lr': improvement / 'low_lr' / 'best.keras',
        'low_lr_intensity_aug': improvement / 'low_lr_intensity_aug' / 'best.keras',
    }
    for file in [stage1_path, *candidates.values()]:
        if not file.is_file():
            raise FileNotFoundError(f'Model file missing: {file}')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    val_paths, val_labels = collect(root / 'dataset', 'validation')
    out = root / 'experiments' / 'two_stage_classification' / datetime.now().strftime('threshold_analysis_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    print('Output:', out, flush=True)
    print('Validation only. No model training, no model modifications, TEST NOT ACCESSED.', flush=True)
    stage1 = tf.keras.models.load_model(stage1_path, compile=False)
    p1 = predict_prob(stage1, val_paths, args.batch_size)
    del stage1
    tf.keras.backend.clear_session()
    if p1.shape != (len(val_labels), 2):
        raise RuntimeError(f'Unexpected Stage 1 output shape: {p1.shape}')
    thresholds = np.round(np.arange(0.30, 0.701, 0.025), 3)
    if 0.5 not in thresholds:
        thresholds = np.sort(np.append(thresholds, 0.5))
    all_rows = []
    summaries = {}
    for name, path in candidates.items():
        print(f'\nEvaluating {name}...', flush=True)
        model = tf.keras.models.load_model(path, compile=False)
        p2 = predict_prob(model, val_paths, args.batch_size)
        del model
        tf.keras.backend.clear_session()
        if p2.shape != (len(val_labels), 2):
            raise RuntimeError(f'Unexpected Stage 2 output shape: {p2.shape}')
        rows = [assess(p1, p2, val_labels, float(t)) for t in thresholds]
        base = min(rows, key=lambda r: abs(r['threshold'] - 0.5))
        best_f1 = max(rows, key=lambda r: (r['macro_f1'], -abs(r['threshold'] - 0.5)))
        best_acc = max(rows, key=lambda r: (r['accuracy'], r['macro_f1'], -abs(r['threshold'] - 0.5)))
        summaries[name] = {'baseline_0_50': base, 'best_macro_f1_exploratory': best_f1,
                           'best_accuracy_exploratory': best_acc, 'checkpoint': str(path)}
        for row in rows:
            all_rows.append({'model': name, **row})
        print(f"  t=0.50: accuracy={base['accuracy']:.4f}, macro-F1={base['macro_f1']:.4f}, viral recall={base['viral_recall']:.4f}")
        print(f"  exploratory max macro-F1: t={best_f1['threshold']:.3f}, accuracy={best_f1['accuracy']:.4f}, macro-F1={best_f1['macro_f1']:.4f}, viral recall={best_f1['viral_recall']:.4f}")
    columns = ['model', 'threshold', 'accuracy', 'macro_f1', 'normal_recall',
               'bacterial_recall', 'viral_recall', 'bacterial_f1', 'viral_f1', 'confusion_matrix']
    with (out / 'threshold_sweep.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**row, 'confusion_matrix': json.dumps(row['confusion_matrix'])})
    payload = {'validation_images': len(val_labels), 'stage1_checkpoint': str(stage1_path),
               'threshold_meaning': 'Original Stage 2 VIRAL probability cutoff, applied via odds adjustment before combining three-class probabilities.',
               'models': summaries,
               'warning': 'Threshold sweep uses validation for selection: maxima are optimistic and are NOT independent performance estimates. No test used. Dataset split has known integrity limitations.'}
    (out / 'threshold_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSaved:', out / 'threshold_summary.json', flush=True)
    print('Saved:', out / 'threshold_sweep.csv', flush=True)


if __name__ == '__main__':
    main()
