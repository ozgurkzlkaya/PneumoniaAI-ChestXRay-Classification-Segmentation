from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
B2 = ROOT / "experiments" / "classification_ablation" / "b2_expanded"
MODEL_PATH = B2 / "best_finetuned.keras"
DATA_PATH = B2 / "dataset" / "validation"
OUTPUT = ROOT / "results" / "classification" / "b2_evaluation"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading B2 model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    paths = []
    labels = []

    for idx, name in enumerate(CLASSES):
        folder = DATA_PATH / name
        if not folder.exists():
            raise FileNotFoundError(folder)

        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(path)
                labels.append(idx)

    if not paths:
        raise RuntimeError("No validation images found.")

    print(f"Validation images: {len(paths)}")

    predictions = []
    probabilities = []

    for i, path in enumerate(paths, 1):
        image = tf.keras.utils.load_img(
            path, target_size=(224, 224), color_mode="rgb"
        )
        array = tf.keras.utils.img_to_array(image)
        batch = np.expand_dims(array, axis=0)

        scores = model.predict(batch, verbose=0)[0]
        predictions.append(int(np.argmax(scores)))
        probabilities.append(scores.tolist())

        if i % 50 == 0:
            print(f"{i}/{len(paths)}")

    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\nB2 VALIDATION RESULTS")
    print(f"Accuracy: {accuracy * 100:.3f}%")
    print(f"Macro F1: {macro_f1:.5f}")
    print("\n", report)
    print("Confusion Matrix:\n", cm)

    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        OUTPUT / "confusion_matrix.csv"
    )

    rows = []
    for path, truth, pred, scores in zip(
        paths, labels, predictions, probabilities
    ):
        rows.append({
            "path": str(path),
            "true_class": CLASSES[truth],
            "predicted_class": CLASSES[pred],
            **{
                f"prob_{name.lower()}": float(scores[j])
                for j, name in enumerate(CLASSES)
            },
        })

    pd.DataFrame(rows).to_csv(
        OUTPUT / "predictions.csv", index=False
    )

    (OUTPUT / "classification_report.txt").write_text(
        f"Accuracy: {accuracy * 100:.3f}%\n"
        f"Macro F1: {macro_f1:.5f}\n\n"
        f"{report}",
        encoding="utf-8",
    )

    print(f"\nResults saved: {OUTPUT}")

if __name__ == "__main__":
    main()
