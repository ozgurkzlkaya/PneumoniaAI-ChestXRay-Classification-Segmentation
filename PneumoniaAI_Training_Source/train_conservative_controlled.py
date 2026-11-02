"""Paired transfer learning + fine-tuning comparison: baseline vs RandomContrast.
Run from any directory; all paths are resolved from the script's project root.
"""
from pathlib import Path
import argparse
import csv
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
INITIAL = ROOT / 'experiments/classification_ablation/configs/shared_initial_model.keras'
TRAIN = ROOT / 'dataset_group_conservative/train'
VAL = ROOT / 'dataset_group_conservative/validation'
OUT = ROOT / 'experiments/conservative_group_training'
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']
WEIGHTS = {0: 4104/(3*1109), 1: 4104/(3*1949), 2: 4104/(3*1046)}
SEED = 42


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)


def datasets():
    common = dict(labels='inferred', label_mode='int', class_names=CLASSES,
                  color_mode='rgb', image_size=(224, 224), batch_size=32)
    train = keras.utils.image_dataset_from_directory(TRAIN, shuffle=True, seed=SEED, **common)
    val = keras.utils.image_dataset_from_directory(VAL, shuffle=False, **common)
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)


def build_model(use_contrast):
    # Loading the SAME initial weights for each branch avoids head-initialization confounding.
    initial = keras.models.load_model(INITIAL, compile=False)
    aug = initial.get_layer('data_augmentation')
    base = initial.get_layer('mobilenetv2_1.00_224')
    base.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = aug(inputs)
    if use_contrast:
        x = layers.RandomContrast(0.1, seed=SEED + 7, name='experimental_random_contrast')(x)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = initial.get_layer('global_average_pooling2d')(x)
    x = initial.get_layer('dropout')(x)
    x = initial.get_layer('dense')(x)
    x = initial.get_layer('dropout_1')(x)
    outputs = initial.get_layer('dense_1')(x)
    return keras.Model(inputs, outputs, name='Pneumonia_MobileNetV2_ContrastExperiment')


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def callbacks(folder, stage, min_lr):
    return [
        keras.callbacks.ModelCheckpoint(str(folder / f'best_{stage}.keras'), monitor='val_loss',
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=2, min_lr=min_lr, verbose=1),
    ]


def fit_stage(model, train, val, folder, stage, epochs, lr, min_lr):
    compile_model(model, lr)
    print(f'\n=== {folder.name}: {stage} ===', flush=True)
    hist = model.fit(train, validation_data=val, epochs=epochs, class_weight=WEIGHTS,
                     callbacks=callbacks(folder, stage, min_lr), verbose=2)
    with (folder / f'history_{stage}.json').open('w', encoding='utf-8') as f:
        json.dump({k: [float(v) for v in values] for k, values in hist.history.items()}, f, indent=2)
    return keras.models.load_model(folder / f'best_{stage}.keras', compile=False)


def evaluate(model, val):
    ys, probs = [], []
    for x, y in val:
        ys.append(y.numpy())
        probs.append(model(x, training=False).numpy())
    y = np.concatenate(ys)
    pred = np.concatenate(probs).argmax(axis=1)
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            'normal_recall': float(rec[0]), 'bacterial_recall': float(rec[1]),
            'viral_recall': float(rec[2]),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['control', 'contrast'], required=True)
    parser.add_argument('--transfer-epochs', type=int, default=15)
    parser.add_argument('--finetune-epochs', type=int, default=12)
    args = parser.parse_args()
    if args.transfer_epochs < 1 or args.finetune_epochs < 1:
        parser.error('Epoch counts must be positive')
    for p in (INITIAL, TRAIN, VAL):
        if not p.exists():
            raise FileNotFoundError(p)
    folder = OUT / args.arm
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Existing experiment output, refusing overwrite: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    set_seed()
    train, val = datasets()
    model = build_model(args.arm == 'contrast')
    model = fit_stage(model, train, val, folder, 'transfer', args.transfer_epochs, 5e-4, 1e-6)
    base = model.get_layer('mobilenetv2_1.00_224')
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model = fit_stage(model, train, val, folder, 'finetuned', args.finetune_epochs, 1e-5, 1e-7)
    result = evaluate(model, val)
    with (folder / 'validation_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('\nFINAL VALIDATION METRICS', json.dumps(result, indent=2), flush=True)
    print('Saved to:', folder, flush=True)


if __name__ == '__main__':
    main()
