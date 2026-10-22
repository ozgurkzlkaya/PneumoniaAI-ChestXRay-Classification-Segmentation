from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']
MODELS = {
    'Baseline CNN': ROOT / 'best_pneumonia_model.keras',
    'MobileNetV2': ROOT / 'best_transfer_model.keras',
    'Fine-Tuned MobileNetV2': ROOT / 'best_finetuned_model.keras',
}


def expected_calibration_error(confidence, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = len(confidence)
    ece = 0.0
    rows = []
    for i in range(bins):
        lower, upper = edges[i], edges[i + 1]
        inside = (confidence >= lower) & ((confidence <= upper) if i == bins - 1 else (confidence < upper))
        n = int(inside.sum())
        if n:
            mean_confidence = float(confidence[inside].mean())
            mean_accuracy = float(correct[inside].mean())
            ece += (n / total) * abs(mean_accuracy - mean_confidence)
        else:
            mean_confidence = float('nan')
            mean_accuracy = float('nan')
        rows.append({'bin_start': lower, 'bin_end': upper, 'count': n,
                     'mean_confidence': mean_confidence, 'observed_accuracy': mean_accuracy})
    return float(ece), pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['validation'], default='validation')
    args = parser.parse_args()
    data_dir = ROOT / 'dataset' / args.split
    out_dir = ROOT / 'experiments' / 'classification_error_analysis' / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in CLASSES:
        if not (data_dir / name).is_dir():
            raise FileNotFoundError(data_dir / name)
    files = sorted((p for name in CLASSES for p in (data_dir / name).iterdir()
                    if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}),
                   key=lambda p: (CLASSES.index(p.parent.name), p.name))
    if not files:
        raise RuntimeError('No validation images found')
    true = np.array([CLASSES.index(p.parent.name) for p in files], dtype=np.int32)
    print(f'Validation images: {len(files)}')

    # Dataset yields images in the same sorted class/file order as `files`.
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, labels='inferred', label_mode='int', class_names=CLASSES,
        color_mode='rgb', image_size=(224, 224), batch_size=32, shuffle=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds_labels = np.concatenate([labels.numpy() for _, labels in ds])
    if not np.array_equal(true, ds_labels):
        raise RuntimeError('Class order mismatch: do not continue')

    summary = []
    for model_name, model_path in MODELS.items():
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        print(f'\nEvaluating {model_name}...')
        model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            raise ValueError(f'Unexpected multiple inputs: {model_name}')
        if tuple(input_shape[1:3]) != (224, 224):
            raise ValueError(f'Unexpected image size {input_shape} for {model_name}; inspect original training preprocessing')
        probs = model.predict(ds, verbose=1)
        if probs.shape != (len(files), len(CLASSES)):
            raise ValueError(f'Unexpected output shape: {probs.shape}')
        if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1, atol=1e-3):
            raise ValueError(f'Invalid probability outputs for {model_name}')
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        correct = pred == true
        ece, calibration = expected_calibration_error(conf, correct)
        onehot = np.eye(len(CLASSES))[true]
        brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
        accuracy = float(accuracy_score(true, pred))
        report = classification_report(true, pred, labels=[0, 1, 2],
                                       target_names=CLASSES, output_dict=True, zero_division=0)
        slug = {'Baseline CNN':'baseline_cnn','MobileNetV2':'mobilenetv2',
                'Fine-Tuned MobileNetV2':'finetuned_mobilenetv2'}[model_name]
        rows = pd.DataFrame({
            'file': [str(p.relative_to(ROOT)) for p in files],
            'actual': [CLASSES[i] for i in true],
            'predicted': [CLASSES[i] for i in pred],
            'correct': correct,
            'confidence': conf,
            **{f'score_{cls}': probs[:, j] for j, cls in enumerate(CLASSES)},
        })
        rows.to_csv(out_dir / f'{slug}_predictions.csv', index=False, encoding='utf-8-sig')
        rows.loc[~rows.correct].sort_values('confidence', ascending=False).to_csv(
            out_dir / f'{slug}_errors.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(confusion_matrix(true, pred, labels=[0, 1, 2]),
                     index=CLASSES, columns=CLASSES).to_csv(out_dir / f'{slug}_confusion_matrix.csv')
        calibration.to_csv(out_dir / f'{slug}_calibration.csv', index=False)
        (out_dir / f'{slug}_classification_report.json').write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        record = {'model': model_name, 'n': len(files), 'accuracy': accuracy,
                  'macro_f1': float(report['macro avg']['f1-score']),
                  'ece_10_bins': ece, 'multiclass_brier': brier,
                  'incorrect': int((~correct).sum()),
                  'mean_confidence_wrong': float(conf[~correct].mean()) if (~correct).any() else None,
                  'high_confidence_wrong_90pct': int(((~correct) & (conf >= .9)).sum()),
                  **{f'recall_{cls}': float(report[cls]['recall']) for cls in CLASSES}}
        summary.append(record)
        print(f"Accuracy: {accuracy:.4f} | Macro F1: {record['macro_f1']:.4f} | "
              f'ECE: {ece:.4f} | Brier: {brier:.4f} | Errors: {record["incorrect"]}')
        print(f'Wrong predictions with score >= 90%: {record["high_confidence_wrong_90pct"]}')
        del model
        tf.keras.backend.clear_session()
    pd.DataFrame(summary).to_csv(out_dir / 'model_summary.csv', index=False, encoding='utf-8-sig')
    print(f'\nReports saved to: {out_dir}')
    print('Note: softmax scores are not calibrated clinical probabilities.')


if __name__ == '__main__':
    main()
