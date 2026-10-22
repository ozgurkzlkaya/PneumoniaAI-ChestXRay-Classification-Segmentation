from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ============================================================
# AYARLAR
# ============================================================

TEST_DIR = Path("dataset") / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = ["NORMAL", "BACTERIAL", "VIRAL"]


# ============================================================
# TEST VERI SETI
# ============================================================

print("\nTest veri seti yukleniyor...\n")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# EN IYI MODELI YUKLE
# ============================================================

print("\nEn iyi model yukleniyor...\n")

model = tf.keras.models.load_model(
    "best_pneumonia_model.keras"
)


# ============================================================
# TEST LOSS VE ACCURACY
# ============================================================

print("\n==============================")
print("TEST SONUCLARI")
print("==============================\n")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : %{test_accuracy * 100:.2f}")


# ============================================================
# GERCEK ETIKETLER
# ============================================================

y_true = np.concatenate(
    [labels.numpy() for images, labels in test_ds]
)


# ============================================================
# TAHMINLER
# ============================================================

print("\nTahminler yapiliyor...\n")

y_prob = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(y_prob, axis=1)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Pneumonia Classification - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nconfusion_matrix.png kaydedildi.")


# ============================================================
# MULTI-CLASS ROC / AUC
# ============================================================

y_true_binary = label_binarize(
    y_true,
    classes=[0, 1, 2]
)

plt.figure(figsize=(8, 6))

auc_scores = {}

for i, class_name in enumerate(CLASS_NAMES):

    fpr, tpr, _ = roc_curve(
        y_true_binary[:, i],
        y_prob[:, i]
    )

    roc_auc = auc(fpr, tpr)

    auc_scores[class_name] = roc_auc

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC = {roc_auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Multi-Class ROC Curves (One-vs-Rest)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.close()


print("\nroc_curve.png kaydedildi.")


# ============================================================
# AUC SONUCLARI
# ============================================================

print("\n==============================")
print("AUC SONUCLARI")
print("==============================")

for class_name, score in auc_scores.items():

    print(
        f"{class_name}: {score:.4f}"
    )


print("\n==============================")
print("DEGERLENDIRME TAMAMLANDI")
print("==============================")