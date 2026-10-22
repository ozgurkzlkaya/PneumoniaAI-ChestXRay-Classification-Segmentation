from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'dataset' / 'validation').is_dir():
    ROOT = Path.cwd()
BASE = ROOT / 'experiments' / 'contrast_augmentation_training'
CSV = BASE / 'robustness_comparison' / 'per_image_predictions.csv'
OUTPUT = BASE / 'robustness_comparison' / 'per_image_probabilities.csv'
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']
SIZE = (224, 224)
BATCH = 32

def load_image(path):
    content = tf.io.read_file(str(ROOT / path))
    img = tf.io.decode_image(content, channels=3, expand_animations=False)
    img.set_shape((None, None, 3))
    return tf.image.resize(img, SIZE, method='bilinear')

def main():
    df = pd.read_csv(CSV)
    if len(df) != 875 or df['file'].duplicated().any():
        raise ValueError('Expected 875 unique images')
    if not df['actual'].isin(CLASSES).all():
        raise ValueError('Unexpected class label')
    files = df['file'].astype(str).str.replace('\\', '/', regex=False).tolist()
    for p in files:
        if not (ROOT / p).is_file():
            raise FileNotFoundError(ROOT / p)
    ds = tf.data.Dataset.from_tensor_slices(files).map(
        lambda p: tf.image.resize(
            tf.io.decode_image(tf.io.read_file(tf.strings.join([str(ROOT).replace('\\', '/') + '/', p])), channels=3, expand_animations=False),
            SIZE, method='bilinear'
        ), num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    out = df[['file', 'actual']].copy()
    for arm in ('control', 'contrast'):
        model_path = BASE / arm / 'best_finetuned.keras'
        print(f'Loading {arm}: {model_path}', flush=True)
        model = keras.models.load_model(model_path, compile=False)
        probs = model.predict(ds, verbose=1)
        if probs.shape != (len(df), 3):
            raise ValueError(f'Unexpected prediction shape: {probs.shape}')
        pred = np.argmax(probs, axis=1)
        expected = df[f'{arm}_original_prediction'].map({v:i for i,v in enumerate(CLASSES)}).to_numpy()
        mismatch = int(np.sum(pred != expected))
        print(f'{arm}: mismatches vs existing predictions = {mismatch}/875', flush=True)
        if mismatch:
            raise RuntimeError('Prediction mismatch: check image decoding/resizing before using probabilities')
        for idx, label in enumerate(CLASSES):
            out[f'{arm}_prob_{label.lower()}'] = probs[:, idx]
        out[f'{arm}_prediction'] = [CLASSES[i] for i in pred]
        out[f'{arm}_confidence'] = np.max(probs, axis=1)
        sorted_probs = np.sort(probs, axis=1)
        out[f'{arm}_margin'] = sorted_probs[:, -1] - sorted_probs[:, -2]
        out[f'{arm}_correct'] = out[f'{arm}_prediction'] == out['actual']
        del model
        keras.backend.clear_session()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(f'Refusing to overwrite: {OUTPUT}')
    out.to_csv(OUTPUT, index=False)
    print(f'Saved {len(out)} rows to {OUTPUT}', flush=True)

if __name__ == '__main__':
    main()
