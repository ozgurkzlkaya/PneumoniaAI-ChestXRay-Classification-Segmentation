"""Step 56: paired validation-only contrast robustness for two saved models.
No training, no test set, no dataset changes. Python 3.12 / TF 2.21.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / 'dataset' / 'validation'
EXP = ROOT / 'experiments' / 'contrast_augmentation_training'
OUT = EXP / 'robustness_comparison'
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']
FACTORS = (1.0, 0.8, 1.2)
EXPECTED_ORIGINAL = {'control': 0.7988571428571428, 'contrast': 0.7977142857142857}

def summarize(y, probs):
    pred = np.argmax(probs, axis=1)
    recall = recall_score(y, pred, labels=[0,1,2], average=None, zero_division=0)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=[0,1,2], average='macro', zero_division=0)),
        'normal_recall': float(recall[0]),
        'bacterial_recall': float(recall[1]),
        'viral_recall': float(recall[2]),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0,1,2]).tolist(),
    }

def main():
    if not VAL.is_dir():
        raise FileNotFoundError(VAL)
    ds = tf.keras.utils.image_dataset_from_directory(
        VAL, labels='inferred', label_mode='int', class_names=CLASSES,
        color_mode='rgb', image_size=(224,224), batch_size=32, shuffle=False)
    paths = [str(Path(p).resolve().relative_to(ROOT.resolve())).replace('\\','/') for p in ds.file_paths]
    y = np.array([int(v) for _, labels in ds for v in labels.numpy()], dtype=np.int32)
    if len(paths) != len(y) or len(y) != 875:
        raise RuntimeError(f'Unexpected validation size: files={len(paths)} labels={len(y)}')
    all_results = {}
    predictions = {}
    per_image = pd.DataFrame({'file': paths, 'actual': [CLASSES[v] for v in y]})
    for arm in ('control', 'contrast'):
        path = EXP / arm / 'best_finetuned.keras'
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f'\n=== {arm.upper()} ===', flush=True)
        model = tf.keras.models.load_model(path, compile=False)
        if tuple(model.input_shape[1:]) != (224,224,3):
            raise RuntimeError(f'Unexpected input shape: {model.input_shape}')
        results = {}
        arm_pred = {}
        for factor in FACTORS:
            if factor == 1.0:
                current_ds = ds
            else:
                # Bind factor at definition time, rather than closing over a loop variable.
                def adjust(images, labels, contrast_factor=factor):
                    images = tf.image.adjust_contrast(images, contrast_factor)
                    return tf.clip_by_value(images, 0.0, 255.0), labels
                current_ds = ds.map(adjust, num_parallel_calls=tf.data.AUTOTUNE)
            probs = model.predict(current_ds.prefetch(tf.data.AUTOTUNE), verbose=0)
            if probs.shape != (len(y), 3) or not np.isfinite(probs).all():
                raise RuntimeError(f'Invalid prediction output for {arm} factor={factor}')
            metric = summarize(y, probs)
            tag = 'original' if factor == 1.0 else ('low' if factor == 0.8 else 'high')
            results[tag] = {'factor': factor, **metric}
            arm_pred[tag] = np.argmax(probs, axis=1)
            per_image[f'{arm}_{tag}_prediction'] = [CLASSES[v] for v in arm_pred[tag]]
            print(f'{tag:8s} acc={metric["accuracy"]:.4f} macro_f1={metric["macro_f1"]:.4f} '
                  f'B_recall={metric["bacterial_recall"]:.4f} V_recall={metric["viral_recall"]:.4f}', flush=True)
            if factor == 1.0 and abs(metric['accuracy'] - EXPECTED_ORIGINAL[arm]) > 1e-9:
                raise RuntimeError(f'BASELINE MISMATCH for {arm}: got {metric["accuracy"]:.8f}, '
                                   f'expected {EXPECTED_ORIGINAL[arm]:.8f}. STOP; check model/loader.')
        baseline = arm_pred['original']
        results['stability'] = {}
        for tag in ('low','high'):
            pred = arm_pred[tag]
            changed = pred != baseline
            fixed = (baseline != y) & (pred == y)
            broken = (baseline == y) & (pred != y)
            details = {
                'changed': int(changed.sum()), 'changed_fraction': float(changed.mean()),
                'wrong_to_correct': int(fixed.sum()), 'correct_to_wrong': int(broken.sum()),
                'net_correct_change': int(fixed.sum() - broken.sum()),
                'transitions_original_to_modified': pd.crosstab(
                    pd.Categorical([CLASSES[v] for v in baseline], categories=CLASSES),
                    pd.Categorical([CLASSES[v] for v in pred], categories=CLASSES),
                    dropna=False).values.tolist(),
            }
            results['stability'][tag] = details
            per_image[f'{arm}_{tag}_changed'] = changed.astype(int)
            per_image[f'{arm}_{tag}_fixed'] = fixed.astype(int)
            per_image[f'{arm}_{tag}_broken'] = broken.astype(int)
            print(f'{tag:8s} changed={details["changed"]} fixed={details["wrong_to_correct"]} '
                  f'broken={details["correct_to_wrong"]} net={details["net_correct_change"]}', flush=True)
        all_results[arm] = results
        predictions[arm] = arm_pred
        del model
    print('\n=== PAIRED COMPARISON ===')
    for tag in ('original','low','high'):
        c = all_results['control'][tag]
        a = all_results['contrast'][tag]
        print(f'{tag:8s} delta_accuracy={a["accuracy"]-c["accuracy"]:+.4f} '
              f'delta_macro_f1={a["macro_f1"]-c["macro_f1"]:+.4f} '
              f'delta_B_recall={a["bacterial_recall"]-c["bacterial_recall"]:+.4f} '
              f'delta_V_recall={a["viral_recall"]-c["viral_recall"]:+.4f}')
    for tag in ('low','high'):
        c = all_results['control']['stability'][tag]['changed']
        a = all_results['contrast']['stability'][tag]['changed']
        print(f'{tag:8s} prediction_changes: control={c}, contrast={a}, delta={a-c:+d}')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    per_image.to_csv(OUT / 'per_image_predictions.csv', index=False, encoding='utf-8-sig')
    print('Saved:', OUT)
    print('NOTE: Exploratory validation comparison; no independent test or statistical significance claim.')

if __name__ == '__main__':
    main()
