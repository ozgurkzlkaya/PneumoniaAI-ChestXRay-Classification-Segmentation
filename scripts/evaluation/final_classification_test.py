
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

BASE = Path(__file__).resolve().parents[2]
TEST_DIR = BASE / "dataset" / "test"
OUT = BASE / "final_classification_report"
OUT.mkdir(exist_ok=True)

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
MODELS = {
    "Baseline CNN": "best_pneumonia_model.keras",
    "MobileNetV2": "best_transfer_model.keras",
    "Fine-Tuned MobileNetV2": "best_finetuned_model.keras",
}

SIZE = (224, 224)
BATCH_SIZE = 16

print("\n=== FINAL SINIFLANDIRMA TESTI ===")

# 1. Dosya listesini olustur
paths = []
true_labels = []

for label, class_name in enumerate(CLASSES):
    folder = TEST_DIR / class_name

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for path in files:
        paths.append(path)
        true_labels.append(label)

y_true = np.array(true_labels, dtype=np.int32)

print(f"Toplam test goruntusu: {len(paths)}")

for i, name in enumerate(CLASSES):
    print(f"{name}: {int(np.sum(y_true == i))}")

if len(paths) != 960:
    raise ValueError("Beklenen 960 test goruntusu bulunamadi.")

# 2. Gorselleri bir defa oku
print("\nGoruntuler hazirlaniyor...")

images = []

for i, path in enumerate(paths, 1):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img = img.resize(SIZE, Image.Resampling.BILINEAR)
        images.append(np.asarray(img, dtype=np.uint8))

X = np.stack(images)

print(f"Veri boyutu: {X.shape}")

# Uygulamadaki gibi ham 0-255 piksel degerleri kullanilir.
# Modelin kendi preprocessing katmani korunur.

records = []
summaries = []
reports = {}
predictions = {}

# 3. Tum modelleri degerlendir
for model_name, filename in MODELS.items():
    print(f"\n=== {model_name} ===")

    model = tf.keras.models.load_model(
        BASE / filename,
        compile=False,
    )

    probs = model.predict(
        X.astype(np.float32),
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    if probs.shape != (len(paths), 3):
        raise ValueError(
            f"{model_name}: Beklenmeyen cikis {probs.shape}"
        )

    if not np.all(np.isfinite(probs)):
        raise ValueError(
            f"{model_name}: Gecersiz tahmin degerleri"
        )

    y_pred = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    predictions[model_name] = y_pred

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    reports[model_name] = report

    summaries.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "normal_recall": report["NORMAL"]["recall"],
        "bacterial_recall": report["BACTERIAL"]["recall"],
        "viral_recall": report["VIRAL"]["recall"],
        "viral_f1": report["VIRAL"]["f1-score"],
        "errors": int(np.sum(y_true != y_pred)),
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Viral Recall: {report['VIRAL']['recall']:.4f}")
    print(f"Viral F1: {report['VIRAL']['f1-score']:.4f}")
    print(f"Hata sayisi: {np.sum(y_true != y_pred)}")

    print("\nSinif bazli rapor:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASSES,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1, 2]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES,
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(model_name)
    plt.tight_layout()

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        OUT / f"confusion_matrix_{safe_name}.png",
        dpi=180,
    )
    plt.close()

    for i, path in enumerate(paths):
        records.append({
            "filename": str(path.relative_to(TEST_DIR)),
            "model": model_name,
            "true_class": CLASSES[y_true[i]],
            "predicted_class": CLASSES[y_pred[i]],
            "correct": bool(y_true[i] == y_pred[i]),
            "confidence": float(confidence[i]),
            "prob_normal": float(probs[i, 0]),
            "prob_bacterial": float(probs[i, 1]),
            "prob_viral": float(probs[i, 2]),
        })

    del model
    tf.keras.backend.clear_session()

# 4. Tum tahminleri kaydet
results_df = pd.DataFrame(records)
results_df.to_csv(
    OUT / "all_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

summary_df = pd.DataFrame(summaries)
summary_df.to_csv(
    OUT / "model_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# 5. Modellerin anlasamadigi goruntuler
prediction_matrix = np.column_stack([
    predictions[name] for name in MODELS
])

disagreement = np.any(
    prediction_matrix != prediction_matrix[:, :1],
    axis=1,
)

disagreement_rows = []

for i in np.where(disagreement)[0]:
    row = {
        "filename": str(paths[i].relative_to(TEST_DIR)),
        "true_class": CLASSES[y_true[i]],
    }

    for model_name in MODELS:
        row[model_name] = CLASSES[
            predictions[model_name][i]
        ]

    disagreement_rows.append(row)

pd.DataFrame(disagreement_rows).to_csv(
    OUT / "model_disagreements.csv",
    index=False,
    encoding="utf-8-sig",
)

# 6. Yuksek skorla yapilan hatalar
high_conf_errors = results_df[
    (~results_df["correct"])
    & (results_df["confidence"] >= 0.90)
].sort_values("confidence", ascending=False)

high_conf_errors.to_csv(
    OUT / "high_confidence_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

# 7. Uc modelin de yanlis bildigi goruntuler
all_wrong = np.ones(len(paths), dtype=bool)

for model_name in MODELS:
    all_wrong &= predictions[model_name] != y_true

shared_errors = []

for i in np.where(all_wrong)[0]:
    row = {
        "filename": str(paths[i].relative_to(TEST_DIR)),
        "true_class": CLASSES[y_true[i]],
    }

    for model_name in MODELS:
        row[model_name] = CLASSES[
            predictions[model_name][i]
        ]

    shared_errors.append(row)

pd.DataFrame(shared_errors).to_csv(
    OUT / "shared_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

# 8. JSON raporu
with open(
    OUT / "classification_reports.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(reports, f, indent=2, ensure_ascii=False)

print("\n=== FINAL KARSILASTIRMA ===")
print(summary_df.to_string(index=False))

print("\nModellerin anlasamadigi goruntu sayisi:",
      int(disagreement.sum()))

print("Uc modelin de yanlis bildigi goruntu sayisi:",
      int(all_wrong.sum()))

print("Yuksek skorlu yanlis tahmin kaydi:",
      len(high_conf_errors))

print("\nSonuclar kaydedildi:")
print(OUT)
