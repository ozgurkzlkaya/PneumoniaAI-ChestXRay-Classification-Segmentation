
from pathlib import Path
import argparse
import csv
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "classification_ablation"
INITIAL_MODEL = EXPERIMENT / "configs" / "shared_initial_model.keras"

CLASS_NAMES = ["NORMAL", "BACTERIAL", "VIRAL"]
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_WEIGHT = {
    0: 1.231,
    1: 0.714,
    2: 1.272
}


def get_directories(mode):
    if mode == "original":
        data_dir = ROOT / "dataset"
    else:
        data_dir = EXPERIMENT / "masked" / "dataset"

    output_dir = EXPERIMENT / mode
    return data_dir, output_dir


def create_datasets(data_dir):
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir / "train",
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir / "validation",
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def save_history(history, path):
    keys = list(history.history.keys())
    rows = zip(*(history.history[key] for key in keys))

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch"] + keys)

        for epoch, values in enumerate(rows, start=1):
            writer.writerow([epoch] + list(values))


def create_callbacks(path, min_lr):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=min_lr,
            verbose=1
        )
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["original", "masked"],
        required=True
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Sadece model ve veri yuklemeyi dogrula"
    )

    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    data_dir, output_dir = get_directories(args.mode)

    if not INITIAL_MODEL.is_file():
        raise FileNotFoundError(INITIAL_MODEL)

    for split in ["train", "validation"]:
        if not (data_dir / split).is_dir():
            raise FileNotFoundError(data_dir / split)

    print(f"\nDENEY: {args.mode.upper()}")
    print(f"Veri: {data_dir}")
    print(f"Model ciktilari: {output_dir}")

    train_ds, val_ds = create_datasets(data_dir)

    model = keras.models.load_model(
        INITIAL_MODEL,
        compile=False
    )

    if args.smoke_test:
        images, labels = next(iter(train_ds))
        predictions = model(images[:2], training=False)

        print("\nSMOKE TEST BASARILI")
        print("Goruntu:", images.shape)
        print("Etiket:", labels.shape)
        print("Tahmin:", predictions.shape)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = output_dir / "best_transfer.keras"
    stage2_path = output_dir / "best_finetuned.keras"

    protected = [
        stage1_path,
        stage2_path,
        output_dir / "transfer_history.csv",
        output_dir / "finetune_history.csv"
    ]

    existing = [
        str(path)
        for path in protected
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Mevcut deney ciktilari korunuyor:\n"
            + "\n".join(existing)
        )

    print("\nASAMA 1: TRANSFER LEARNING")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=5e-4
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=CLASS_WEIGHT,
        callbacks=create_callbacks(
            stage1_path,
            1e-6
        )
    )

    save_history(
        history1,
        output_dir / "transfer_history.csv"
    )

    print("\nASAMA 2: FINE-TUNING")

    model = keras.models.load_model(
        stage1_path,
        compile=False
    )

    base_model = model.get_layer(
        "mobilenetv2_1.00_224"
    )

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(
            layer,
            keras.layers.BatchNormalization
        ):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=12,
        class_weight=CLASS_WEIGHT,
        callbacks=create_callbacks(
            stage2_path,
            1e-7
        )
    )

    save_history(
        history2,
        output_dir / "finetune_history.csv"
    )

    print("\nDENEY TAMAMLANDI")
    print("Transfer modeli:", stage1_path)
    print("Fine-tuned model:", stage2_path)


if __name__ == "__main__":
    main()
