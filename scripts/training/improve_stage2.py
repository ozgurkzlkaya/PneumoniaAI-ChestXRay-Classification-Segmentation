"""Stage 2 controlled follow-up. Keeps Stage 1 fixed, never reads test."""
import argparse, csv, json
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from train_two_stage import collect, make_dataset, predict_prob, CLASSES, weights, compile_model


def stage2_dataset(paths, labels, batch, training=False, augment=False, seed=42):
    keep = labels != 0
    paths, labels = paths[keep], (labels[keep] == 2).astype('int32')
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(tf.cast(image, tf.float32), (224, 224))
        if augment:
            # Subtle intensity augmentation, applied to training only.
            image = tf.image.random_brightness(image, max_delta=7.0)
            image = tf.image.random_contrast(image, lower=0.94, upper=1.06)
            image = tf.clip_by_value(image, 0., 255.)
        return image, label
    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE), labels


def run_variant(name, checkpoint, train_paths, train_labels, val_paths, val_labels, out, args, augment):
    tf.keras.utils.set_random_seed(args.seed)
    model = tf.keras.models.load_model(checkpoint)
    backbone = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                     and layer.name.startswith('mobilenetv2')), None)
    if backbone is None:
        raise RuntimeError('MobileNetV2 backbone not found in checkpoint')
    backbone.trainable = True
    for layer in backbone.layers[:-args.unfreeze_last]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, args.learning_rate)
    train_ds, y = stage2_dataset(train_paths, train_labels, args.batch_size, True, augment, args.seed)
    val_ds, _ = stage2_dataset(val_paths, val_labels, args.batch_size)
    folder = out / name
    folder.mkdir()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(folder / 'best.keras'), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=.5, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(str(folder / 'history.csv')),
    ]
    print(f'\n=== {name} (Stage 1 frozen; Stage 2 fine-tuning) ===', flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=weights(y), callbacks=callbacks)
    del model
    tf.keras.backend.clear_session()
    return folder / 'best.keras'


def evaluate(stage1, stage2, val_paths, val_labels, batch):
    p1 = predict_prob(stage1, val_paths, batch)
    p2 = predict_prob(stage2, val_paths, batch)
    probs = np.stack([p1[:, 0], p1[:, 1]*p2[:, 0], p1[:, 1]*p2[:, 1]], axis=1)
    pred = probs.argmax(axis=1)
    return {'accuracy': float(accuracy_score(val_labels, pred)),
            'macro_f1': float(f1_score(val_labels, pred, average='macro')),
            'confusion_matrix': confusion_matrix(val_labels, pred, labels=[0,1,2]).tolist(),
            'classification_report': classification_report(val_labels, pred, labels=[0,1,2],
                target_names=CLASSES, output_dict=True, zero_division=0)}, pred, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--previous-run', required=True, help='Path to existing run_... folder')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--unfreeze-last', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    previous = Path(args.previous_run).resolve()
    stage1_file = previous / 'stage1_selected.keras'
    stage2_file = previous / 'stage2_selected.keras'
    for p in (stage1_file, stage2_file):
        if not p.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {p}')
    train_paths, train_labels = collect(root / 'dataset', 'train')
    val_paths, val_labels = collect(root / 'dataset', 'validation')
    out = root / 'experiments' / 'two_stage_classification' / datetime.now().strftime('stage2_improvement_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps({**vars(args), 'stage1_checkpoint': str(stage1_file),
        'stage2_checkpoint': str(stage2_file), 'test_used': False}, indent=2), encoding='utf-8')
    print('Output:', out, '\nStage 1 checkpoint unchanged:', stage1_file, '\nTEST SET NOT ACCESSED.', flush=True)
    stage1 = tf.keras.models.load_model(stage1_file)
    baseline = tf.keras.models.load_model(stage2_file)
    results = {}
    for name, model in [('baseline', baseline)]:
        score, _, _ = evaluate(stage1, model, val_paths, val_labels, args.batch_size)
        results[name] = score
        print(name, 'accuracy', score['accuracy'], 'macro_f1', score['macro_f1'], flush=True)
    del baseline
    for name, aug in [('low_lr', False), ('low_lr_intensity_aug', True)]:
        best_file = run_variant(name, stage2_file, train_paths, train_labels,
                                val_paths, val_labels, out, args, aug)
        candidate = tf.keras.models.load_model(best_file)
        score, pred, probs = evaluate(stage1, candidate, val_paths, val_labels, args.batch_size)
        results[name] = {**score, 'checkpoint': str(best_file)}
        with (out / name / 'validation_predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'true_class', 'predicted_class', 'p_normal', 'p_bacterial', 'p_viral'])
            for path, y, p, prob in zip(val_paths, val_labels, pred, probs):
                writer.writerow([str(Path(path).relative_to(root)), CLASSES[y], CLASSES[p], *map(float, prob)])
        print(name, 'accuracy', score['accuracy'], 'macro_f1', score['macro_f1'], flush=True)
        del candidate
    winner = max(results, key=lambda name: (results[name]['macro_f1'], results[name]['accuracy']))
    (out / 'comparison.json').write_text(json.dumps({'results': results, 'best_by_validation_macro_f1': winner,
        'note': 'Exploratory validation; prior split leakage and repeated validation selection. No test used.'}, indent=2), encoding='utf-8')
    print('\nBEST BY VALIDATION MACRO-F1:', winner, '\nResults:', out, flush=True)

if __name__ == '__main__':
    main()
