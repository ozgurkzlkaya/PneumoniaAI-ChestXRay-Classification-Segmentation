
from pathlib import Path
import csv

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "classification_ablation"
REPORTS = EXP / "reports"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
BATCH_SIZE = 32
IMG_SIZE = (224, 224)


def evaluate(mode):
    if mode == "original":
        data_dir = ROOT / "dataset" / "validation"
    else:
        data_dir = EXP / "masked" / "dataset" / "validation"

    model_path = EXP / mode / "best_finetuned.keras"

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    print(f"\nDEGERLENDIRME: {mode.upper()}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    y_true = []
    y_pred = []

    for images, labels in dataset:
        probabilities = model(images, training=False).numpy()
        predictions = np.argmax(probabilities, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    print("\n", report)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )

    print("Confusion Matrix:")
    print(matrix)

    REPORTS.mkdir(parents=True, exist_ok=True)

    with (REPORTS / f"{mode}_classification_report.txt").open(
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"Accuracy: {accuracy:.6f}\n")
        file.write(f"Macro F1: {macro_f1:.6f}\n\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(matrix))

    with (REPORTS / f"{mode}_confusion_matrix.csv").open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted"] + CLASSES)

        for class_name, row in zip(CLASSES, matrix):
            writer.writerow([class_name] + row.tolist())

    return accuracy, macro_f1


def main():
    results = {}

    for mode in ["original", "masked"]:
        results[mode] = evaluate(mode)

    print("\nKARSILASTIRMA")
    print(
        f"Original Accuracy: {results['original'][0]:.4f}"
    )
    print(
        f"Masked Accuracy:   {results['masked'][0]:.4f}"
    )
    print(
        f"Original Macro F1: {results['original'][1]:.4f}"
    )
    print(
        f"Masked Macro F1:   {results['masked'][1]:.4f}"
    )


if __name__ == "__main__":
    main()
