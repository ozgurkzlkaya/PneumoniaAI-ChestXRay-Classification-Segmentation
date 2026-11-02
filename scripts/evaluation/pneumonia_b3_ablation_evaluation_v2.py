"""Compare saved A/B2/B3 models on their matched validation sets.
Read-only for datasets and models; refuses to overwrite existing output directory.
Uses TensorFlow PNG/JPEG decoding and bilinear resize, matching Keras image_dataset_from_directory.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def read_image(path):
    data = tf.io.read_file(str(path))
    # Keras image_dataset_from_directory uses tf.io.decode_image(expand_animations=False).
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    return tf.image.resize(image, (224, 224), method='bilinear')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    root = args.root.resolve()
    exp = root / 'experiments' / 'classification_ablation'
    specs = [
        ('A_original', exp / 'original' / 'best_finetuned.keras', root / 'dataset' / 'validation'),
        ('B2_black', exp / 'b2_expanded' / 'best_finetuned.keras', exp / 'b2_expanded' / 'dataset' / 'validation'),
        ('B3_alpha025', exp / 'b3_alpha025' / 'best_finetuned.keras', exp / 'b3_alpha025' / 'dataset' / 'validation'),
        ('B3_alpha050', exp / 'b3_alpha050' / 'best_finetuned.keras', exp / 'b3_alpha050' / 'dataset' / 'validation'),
    ]
    out = root / 'results' / 'classification' / 'ablation_uniform_evaluation'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing results: {out}')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    for name, model_path, data_path in specs:
        if not model_path.is_file():
            raise FileNotFoundError(f'{name}: missing model: {model_path}')
        for cls in CLASSES:
            if not (data_path / cls).is_dir():
                raise FileNotFoundError(f'{name}: missing class directory: {data_path / cls}')
    manifests = []
    for name, model_path, data_path in specs:
        paths = []
        labels = []
        for i, cls in enumerate(CLASSES):
            items = sorted(p for p in (data_path / cls).rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
            paths.extend(items)
            labels.extend([i] * len(items))
        if not paths:
            raise ValueError(f'{name}: no images')
        manifests.append((name, model_path, data_path, paths, np.array(labels, dtype=np.int64)))
    expected = [(CLASSES[int(label)], p.stem) for p, label in zip(manifests[0][3], manifests[0][4])]
    for name, _, _, paths, labels in manifests[1:]:
        actual = [(CLASSES[int(label)], p.stem) for p, label in zip(paths, labels)]
        if actual != expected:
            raise ValueError(f'{name}: validation file identities/order differ from A; refusing comparison')
    out.mkdir(parents=True, exist_ok=False)
    summary_rows = []
    for name, model_path, data_path, paths, y_true in manifests:
        print(f'\n=== {name}: {len(paths)} images ===', flush=True)
        model = tf.keras.models.load_model(model_path, compile=False)
        probs = []
        for start in range(0, len(paths), args.batch_size):
            batch = tf.stack([read_image(p) for p in paths[start:start + args.batch_size]])
            scores = np.asarray(model(batch, training=False).numpy())
            if scores.ndim != 2 or scores.shape[1] != 3:
                raise ValueError(f'{name}: unexpected prediction shape: {scores.shape}')
            probs.extend(scores.tolist())
            if (start // args.batch_size + 1) % 5 == 0:
                print(f'  {min(start + args.batch_size, len(paths))}/{len(paths)}', flush=True)
        probs = np.asarray(probs)
        y_pred = np.argmax(probs, axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        macro = f1_score(y_true, y_pred, average='macro')
        row = {'model': name, 'n': len(paths), 'accuracy_pct': round(100 * acc, 4), 'macro_f1': round(float(macro), 6)}
        for i, cls in enumerate(CLASSES):
            row[f'{cls.lower()}_precision'] = round(float(precision[i]), 6)
            row[f'{cls.lower()}_recall'] = round(float(recall[i]), 6)
            row[f'{cls.lower()}_f1'] = round(float(f1[i]), 6)
            row[f'{cls.lower()}_support'] = int(support[i])
        summary_rows.append(row)
        with (out / f'{name}_predictions.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['path', 'true_class', 'predicted_class', 'prob_normal', 'prob_bacterial', 'prob_viral'])
            for p, true, pred, score in zip(paths, y_true, y_pred, probs):
                writer.writerow([str(p), CLASSES[int(true)], CLASSES[int(pred)], *map(float, score)])
        with (out / f'{name}_confusion_matrix.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['true\\pred', *CLASSES])
            for cls, counts in zip(CLASSES, cm):
                writer.writerow([cls, *counts.tolist()])
        report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASSES, digits=4, zero_division=0)
        (out / f'{name}_report.txt').write_text(f'Accuracy: {acc:.6f}\nMacro F1: {macro:.6f}\n\n{report}\n', encoding='utf-8')
        print(f'Accuracy: {100 * acc:.3f}% | Macro F1: {macro:.5f}', flush=True)
        print('Recalls: ' + ', '.join(f'{cls}={100 * recall[i]:.2f}%' for i, cls in enumerate(CLASSES)), flush=True)
        print('Confusion matrix:\n' + str(cm), flush=True)
        del model
        tf.keras.backend.clear_session()
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'\nAll results saved to: {out}', flush=True)

if __name__ == '__main__':
    main()
