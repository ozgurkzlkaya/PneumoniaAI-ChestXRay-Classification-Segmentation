
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/conservative_group_training"
SOURCE = EXP / "control/test_probabilities.npz"
OUT = EXP / "figures"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
COLORS = ["#16A085", "#D18B35", "#5677C8"]

names = [
    "roc_curves",
    "precision_recall_curves",
    "confusion_matrix"
]

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

OUT.mkdir(parents=True, exist_ok=True)

for name in names:
    for ext in ["png", "pdf"]:
        path = OUT / f"{name}.{ext}"
        if path.exists():
            raise FileExistsError(f"Dosya zaten mevcut: {path}")

with np.load(SOURCE) as data:
    y_true = data["y_true"]
    y_prob = data["y_prob"]

assert y_prob.shape == (len(y_true), 3)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "savefig.dpi": 300
})

def save_figure(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close()

# ROC curves
plt.figure(figsize=(8, 6))

for i, cls in enumerate(CLASSES):
    binary = (y_true == i).astype(int)
    fpr, tpr, _ = roc_curve(binary, y_prob[:, i])
    score = auc(fpr, tpr)

    plt.plot(
        fpr, tpr,
        color=COLORS[i],
        linewidth=2,
        label=f"{cls} (AUC = {score:.4f})"
    )

plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves")
plt.legend(loc="lower right")
plt.grid(alpha=0.2)

save_figure("roc_curves")

# Precision-Recall curves
plt.figure(figsize=(8, 6))

for i, cls in enumerate(CLASSES):
    binary = (y_true == i).astype(int)

    precision, recall, _ = precision_recall_curve(
        binary, y_prob[:, i]
    )

    ap = average_precision_score(binary, y_prob[:, i])

    plt.plot(
        recall, precision,
        color=COLORS[i],
        linewidth=2,
        label=f"{cls} (AP = {ap:.4f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("One-vs-Rest Precision-Recall Curves")
plt.legend(loc="lower left")
plt.grid(alpha=0.2)

save_figure("precision_recall_curves")

# Confusion matrix
y_pred = y_prob.argmax(axis=1)

cm = confusion_matrix(
    y_true, y_pred, labels=[0, 1, 2]
)

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Test Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

save_figure("confusion_matrix")

print("\n=== BILIMSEL GRAFIKLER HAZIR ===")

for name in names:
    for ext in ["png", "pdf"]:
        path = OUT / f"{name}.{ext}"
        print(f"{path} ({path.stat().st_size:,} bytes)")

print("\nAdim 99 tamamlandi.")
