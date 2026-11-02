from pathlib import Path
import csv
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results/classification/ablation_uniform_evaluation"
OUT = ROOT / "results/classification/decision_boundary_validation"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
COLS = ["prob_normal", "prob_bacterial", "prob_viral"]

def load(filename):
    result = {}
    with (BASE / filename).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["true_class"], Path(row["path"]).stem)
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = np.array([float(row[c]) for c in COLS])
    return result

a = load("A_original_predictions.csv")
b = load("B3_alpha050_predictions.csv")

assert set(a) == set(b), "Prediction files do not match"
keys = sorted(a)
assert len(keys) == 875

y = np.array([CLASSES.index(k[0]) for k in keys])
pa = np.stack([a[k] for k in keys])
pb = np.stack([b[k] for k in keys])
p = 0.5 * pa + 0.5 * pb

rows = []

# Fixed, small BACTERIAL-vs-VIRAL decision adjustments.
# Positive bias favors BACTERIAL; negative favors VIRAL.
for bias in [-0.04, -0.02, 0.0, 0.02, 0.04]:
    scores = p.copy()
    scores[:, 1] += bias / 2
    scores[:, 2] -= bias / 2
    pred = scores.argmax(axis=1)

    row = {
        "bacterial_bias": bias,
        "correct": int((pred == y).sum()),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "normal_recall": recall_score(y, pred, labels=[0], average="macro"),
        "bacterial_recall": recall_score(y, pred, labels=[1], average="macro"),
        "viral_recall": recall_score(y, pred, labels=[2], average="macro"),
    }
    rows.append(row)

print("\nFIXED DECISION BIAS EXPERIMENT - VALIDATION ONLY")
for r in rows:
    print(
        f"bias={r['bacterial_bias']:+.2f} "
        f"correct={r['correct']}/875 "
        f"accuracy={r['accuracy']:.4%} "
        f"macro_f1={r['macro_f1']:.4f} "
        f"bacterial_recall={r['bacterial_recall']:.4%} "
        f"viral_recall={r['viral_recall']:.4%}"
    )

print("\nENSEMBLE MARGIN ANALYSIS")
pred = p.argmax(axis=1)
margin = np.abs(p[:, 1] - p[:, 2])

for cls in [1, 2]:
    subset = (y == cls)
    print(f"\n{CLASSES[cls]}:")
    for low, high in [(0, .05), (.05, .10), (.10, .20), (.20, 1.01)]:
        selected = subset & (margin >= low) & (margin < high)
        n = int(selected.sum())
        errors = int((pred[selected] != y[selected]).sum())
        print(f"  margin [{low:.2f}, {high:.2f}): n={n}, errors={errors}")

with (OUT / "bias_results.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", OUT)
print("No training. No test evaluation.")
