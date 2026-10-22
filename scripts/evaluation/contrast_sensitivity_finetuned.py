"""Step 44: validation-only contrast sensitivity for saved fine-tuned MobileNetV2.
Uses identical tf.keras image_dataset_from_directory loader to historical evaluation.
No retraining, no test set, no dataset modification.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / 'dataset' / 'validation'
MODEL = ROOT / 'best_finetuned_model.keras'
REFERENCE = ROOT / 'experiments/classification_error_analysis/validation/shared_error_analysis/all_image_comparisons.csv'
OUT = ROOT / 'experiments/classification_error_analysis/validation/contrast_sensitivity_finetuned'
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']
FACTORS = [1.0, 0.8, 1.2]
BATCH = 32


def key(path):
    p = Path(str(path))
    try:
        return str(p.resolve().relative_to(ROOT.resolve())).replace('\\', '/').casefold()
    except ValueError:
        return p.name.casefold()


def metrics(y, p):
    pred = np.argmax(p, axis=1)
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            'normal_recall': float(rec[0]), 'bacterial_recall': float(rec[1]),
            'viral_recall': float(rec[2]),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def main():
    for path in (VAL, MODEL, REFERENCE):
        if not path.exists():
            raise FileNotFoundError(path)
    ref = pd.read_csv(REFERENCE)
    for col in ('file', 'actual', 'pred_finetuned'):
        if col not in ref.columns:
            raise ValueError(f'Missing reference column: {col}')
    ds = tf.keras.utils.image_dataset_from_directory(
        VAL, labels='inferred', label_mode='int', class_names=CLASSES,
        color_mode='rgb', image_size=(224, 224), batch_size=BATCH, shuffle=False)
    paths = [key(p) for p in ds.file_paths]
    if len(paths) != len(set(paths)):
        raise RuntimeError('Duplicate file paths in validation loader')
    lookup = {key(r.file): r for r in ref.itertuples(index=False)}
    if len(lookup) != len(ref) or set(paths) != set(lookup):
        missing = list(set(paths) - set(lookup))[:3]
        extra = list(set(lookup) - set(paths))[:3]
        raise RuntimeError(f'CSV and dataset file paths differ; missing={missing}, extra={extra}')
    y = np.array([CLASSES.index(str(lookup[p].actual).upper()) for p in paths], dtype=np.int32)
    historical = np.array([CLASSES.index(str(lookup[p].pred_finetuned).upper()) for p in paths], dtype=np.int32)
    model = tf.keras.models.load_model(MODEL, compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise RuntimeError(f'Unexpected model input: {model.input_shape}')
    results = {}
    predictions = {}
    for factor in FACTORS:
        # Contrast about per-image mean, matching tf.image.adjust_contrast.
        # Input remains 0..255, exactly as in original evaluation.
        if factor == 1.0:
            condition_ds = ds
        else:
            condition_ds = ds.map(
                lambda images, labels: (
                    tf.clip_by_value(tf.image.adjust_contrast(images, factor), 0.0, 255.0), labels),
                num_parallel_calls=tf.data.AUTOTUNE)
        probs = model.predict(condition_ds.prefetch(tf.data.AUTOTUNE), verbose=1)
        if probs.shape != (len(paths), 3) or not np.isfinite(probs).all():
            raise RuntimeError(f'Invalid predictions for factor={factor}')
        predictions[factor] = probs
        results[str(factor)] = metrics(y, probs)
        print(f"factor={factor:.1f} accuracy={results[str(factor)]['accuracy']:.4f} "
              f"macro_f1={results[str(factor)]['macro_f1']:.4f} "
              f"B_recall={results[str(factor)]['bacterial_recall']:.4f} "
              f"V_recall={results[str(factor)]['viral_recall']:.4f}")
        if factor == 1.0:
            mismatches = int(np.sum(np.argmax(probs, axis=1) != historical))
            print(f'BASELINE CHECK: predictions differing from historical CSV: {mismatches}/{len(paths)}')
            if mismatches:
                print('STOP: baseline differs from previous evaluation; do not interpret contrast results.')
                return
    base_pred = np.argmax(predictions[1.0], axis=1)
    records = []
    for i, p in enumerate(paths):
        row = {'file': p, 'actual': CLASSES[y[i]], 'pred_original': CLASSES[base_pred[i]]}
        for factor in (0.8, 1.2):
            pred = int(np.argmax(predictions[factor][i]))
            tag = 'low' if factor == 0.8 else 'high'
            row[f'pred_{tag}'] = CLASSES[pred]
            row[f'changed_{tag}'] = int(pred != base_pred[i])
            row[f'original_correct'] = int(base_pred[i] == y[i])
            row[f'correct_{tag}'] = int(pred == y[i])
            row[f'viral_probability_{tag}'] = float(predictions[factor][i, 2])
        row['viral_probability_original'] = float(predictions[1.0][i, 2])
        records.append(row)
    out = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'per_image_contrast_changes.csv', index=False, encoding='utf-8-sig')
    with (OUT / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    for tag in ('low', 'high'):
        changed = int(out[f'changed_{tag}'].sum())
        corrected = int(((out['original_correct'] == 0) & (out[f'correct_{tag}'] == 1)).sum())
        broken = int(((out['original_correct'] == 1) & (out[f'correct_{tag}'] == 0)).sum())
        print(f'{tag}: changed={changed}, wrong_to_correct={corrected}, correct_to_wrong={broken}')
    print('Saved:', OUT)
    print('NOTE: validation-only sensitivity, not independent performance improvement or causal proof.')


if __name__ == '__main__':
    main()
