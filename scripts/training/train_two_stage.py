"""PneumoniaAI experimental two-stage MobileNetV2 classifier.
Run from project root: python train_two_stage.py --root .
Validation is for selection; test is not evaluated by this script.
"""
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASSES = ('NORMAL', 'BACTERIAL', 'VIRAL')
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def collect(root, split):
    paths, labels = [], []
    for i, name in enumerate(CLASSES):
        folder = root / split / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                paths.append(str(p.resolve()))
                labels.append(i)
    if not paths:
        raise ValueError(f'No images in {root / split}')
    return np.asarray(paths, dtype=str), np.asarray(labels, dtype=np.int32)


def make_dataset(paths, labels, task, batch_size, training=False, seed=42):
    if task == 'stage1':
        y = (labels != 0).astype('int32')
    else:
        keep = labels != 0
        paths, labels = paths[keep], labels[keep]
        y = (labels == 2).astype('int32')  # BACTERIAL=0, VIRAL=1
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(tf.cast(image, tf.float32), IMG_SIZE)
        return image, label

    ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE), y


def build_model(alpha=1.0):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.02),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomTranslation(0.03, 0.03),
    ], name='augmentation')
    x = aug(inputs)
    x = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)(x)
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3), alpha=alpha
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs), backbone


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )


def callbacks(folder, name):
    return [
        tf.keras.callbacks.ModelCheckpoint(str(folder / f'{name}_best.keras'),
                                           monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=2, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(str(folder / f'{name}_history.csv')),
    ]


def weights(y):
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f'Empty class: {counts}')
    return {i: float(len(y) / (2 * counts[i])) for i in range(2)}


def train_one(task, train_paths, train_labels, val_paths, val_labels, args, output):
    train_ds, y_train = make_dataset(train_paths, train_labels, task, args.batch_size, True, args.seed)
    val_ds, _ = make_dataset(val_paths, val_labels, task, args.batch_size)
    model, backbone = build_model(args.alpha)
    compile_model(model, 5e-4)
    print(f'\n=== {task}: frozen backbone ===', flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
              class_weight=weights(y_train), callbacks=callbacks(output, f'{task}_head'))
    model = tf.keras.models.load_model(output / f'{task}_head_best.keras')
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                    and layer.name.startswith('mobilenetv2'))
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    print(f'\n=== {task}: fine tuning ===', flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
              class_weight=weights(y_train), callbacks=callbacks(output, f'{task}_finetuned'))
    # Compare head and finetune checkpoints by validation loss; do not assume finetune wins.
    candidates = [output / f'{task}_head_best.keras', output / f'{task}_finetuned_best.keras']
    evaluated = []
    for path in candidates:
        candidate = tf.keras.models.load_model(path)
        loss, acc = candidate.evaluate(val_ds, verbose=0)
        evaluated.append((float(loss), float(acc), path))
    best = min(evaluated, key=lambda t: t[0])
    chosen = tf.keras.models.load_model(best[2])
    chosen.save(output / f'{task}_selected.keras')
    print(f'{task} selected: {best[2].name} val_loss={best[0]:.4f} val_acc={best[1]:.4f}')
    return chosen, {'head': evaluated[0][:2], 'finetuned': evaluated[1][:2],
                    'selected': best[2].name}


def predict_prob(model, paths, batch_size):
    dummy = np.zeros(len(paths), dtype=np.int32)
    ds, _ = make_dataset(paths, dummy, 'stage1', batch_size)
    return model.predict(ds, verbose=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--head-epochs', type=int, default=15)
    ap.add_argument('--finetune-epochs', type=int, default=12)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.root).resolve()
    dataset = root / 'dataset'
    train_paths, train_labels = collect(dataset, 'train')
    val_paths, val_labels = collect(dataset, 'validation')
    out = root / 'experiments' / 'two_stage_classification' / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    print(f'Output: {out}')
    print('WARNING: Existing split may contain duplicates. Experimental comparison only.')
    print('Test set is NOT accessed.')
    m1, info1 = train_one('stage1', train_paths, train_labels, val_paths, val_labels, args, out)
    m2, info2 = train_one('stage2', train_paths, train_labels, val_paths, val_labels, args, out)
    p1 = predict_prob(m1, val_paths, args.batch_size)
    p2 = predict_prob(m2, val_paths, args.batch_size)
    combined = np.stack([p1[:, 0], p1[:, 1] * p2[:, 0], p1[:, 1] * p2[:, 1]], axis=1)
    predictions = combined.argmax(axis=1)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    report = classification_report(val_labels, predictions, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    result = {'validation_accuracy': float(accuracy_score(val_labels, predictions)),
              'validation_macro_f1': float(f1_score(val_labels, predictions, average='macro')),
              'confusion_matrix': confusion_matrix(val_labels, predictions).tolist(),
              'classification_report': report, 'stage1': info1, 'stage2': info2,
              'caveat': 'Experimental validation only; known split leakage risk and no untouched test.'}
    (out / 'validation_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out / 'validation_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_class', 'predicted_class', 'p_normal', 'p_bacterial', 'p_viral'])
        for path, y, pred, prob in zip(val_paths, val_labels, predictions, combined):
            writer.writerow([str(Path(path).relative_to(root)), CLASSES[y], CLASSES[pred], *map(float, prob)])
    print('\nVALIDATION RESULTS')
    print(json.dumps({'accuracy': result['validation_accuracy'], 'macro_f1': result['validation_macro_f1'],
                      'confusion_matrix': result['confusion_matrix']}, indent=2))
    print(f'Results saved: {out}')


if __name__ == '__main__':
    main()
