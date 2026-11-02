"""Train B3 alpha=0.25 or 0.50 with the B2 matched MobileNetV2 protocol.
Run from any directory. Never overwrites existing model/history outputs.
"""
from pathlib import Path
import argparse
import csv
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT_DEFAULT = Path(__file__).resolve().parent
CLASS_NAMES = ['NORMAL', 'BACTERIAL', 'VIRAL']
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_WEIGHT = {0: 1.231, 1: 0.714, 2: 1.272}
MODES = ('b3_alpha025', 'b3_alpha050')


def create_datasets(data_dir):
    kwargs = dict(labels='inferred', label_mode='int', class_names=CLASS_NAMES,
                  color_mode='rgb', image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir / 'train', shuffle=True, seed=SEED, **kwargs)
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir / 'validation', shuffle=False, **kwargs)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def save_history(history, path):
    keys = list(history.history.keys())
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch'] + keys)
        for epoch, values in enumerate(zip(*(history.history[k] for k in keys)), start=1):
            writer.writerow([epoch] + list(values))


def callbacks(checkpoint, min_lr):
    return [
        keras.callbacks.ModelCheckpoint(filepath=checkpoint, monitor='val_loss',
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=2, min_lr=min_lr, verbose=1),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    root = args.root.resolve()
    base = root / 'experiments' / 'classification_ablation'
    data_dir = base / args.mode / 'dataset'
    output_dir = base / args.mode
    initial_model = base / 'configs' / 'shared_initial_model.keras'

    for path in [initial_model, data_dir / 'train', data_dir / 'validation']:
        if not path.exists():
            raise FileNotFoundError(path)
    if not initial_model.is_file():
        raise ValueError(f'Initial model is not a file: {initial_model}')

    protected = [output_dir / name for name in (
        'best_transfer.keras', 'best_finetuned.keras',
        'transfer_history.csv', 'finetune_history.csv')]
    if not args.smoke_test:
        existing = [str(p) for p in protected if p.exists()]
        if existing:
            raise FileExistsError('Existing B3 outputs are protected:\n' + '\n'.join(existing))

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    print(f'EXPERIMENT: {args.mode}', flush=True)
    print(f'DATA: {data_dir}', flush=True)
    print(f'OUTPUT: {output_dir}', flush=True)
    train_ds, val_ds = create_datasets(data_dir)
    model = keras.models.load_model(initial_model, compile=False)

    if args.smoke_test:
        images, labels = next(iter(train_ds))
        predictions = model(images[:2], training=False)
        print('SMOKE TEST OK')
        print('images:', images.shape, 'labels:', labels.shape,
              'predictions:', predictions.shape)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    transfer_path = output_dir / 'best_transfer.keras'
    finetune_path = output_dir / 'best_finetuned.keras'

    print('STAGE 1: TRANSFER LEARNING', flush=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=15,
                         class_weight=CLASS_WEIGHT,
                         callbacks=callbacks(transfer_path, 1e-6))
    save_history(history1, output_dir / 'transfer_history.csv')

    print('STAGE 2: FINE-TUNING', flush=True)
    model = keras.models.load_model(transfer_path, compile=False)
    base_model = model.get_layer('mobilenetv2_1.00_224')
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=12,
                         class_weight=CLASS_WEIGHT,
                         callbacks=callbacks(finetune_path, 1e-7))
    save_history(history2, output_dir / 'finetune_history.csv')
    print('DONE:', args.mode, flush=True)
    print('Transfer:', transfer_path)
    print('Fine-tuned:', finetune_path)


if __name__ == '__main__':
    main()
