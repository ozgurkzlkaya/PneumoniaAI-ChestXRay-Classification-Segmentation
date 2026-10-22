
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

BASE = Path(__file__).resolve().parents[2]
REPORT = BASE / "final_classification_report"

df = pd.read_csv(REPORT / "all_predictions.csv")

LEAKED_TEST_FILE = (
    "NORMAL/NORMAL2-IM-0173-0001-0002.jpeg"
)

# Windows ve Unix yol ayraclarini esitle
df["normalized_filename"] = (
    df["filename"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
)

clean = df[
    df["normalized_filename"] != LEAKED_TEST_FILE
].copy()

removed = len(df) - len(clean)

print("\n=== KESIN KOPYA HARIC TEST ===")
print("Cikarilan tahmin kaydi:", removed)

if removed != 3:
    raise ValueError(
        "Beklenen 3 model tahmin kaydi cikarilamadi."
    )

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]

results = []

for model_name, group in clean.groupby("model"):
    y_true = group["true_class"]
    y_pred = group["predicted_class"]

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    results.append({
        "model": model_name,
        "test_count": len(group),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "viral_recall": report["VIRAL"]["recall"],
        "viral_f1": report["VIRAL"]["f1-score"],
    })

summary = pd.DataFrame(results)

summary.to_csv(
    REPORT / "clean_test_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print(summary.to_string(index=False))
print("\nSonuc kaydedildi.")
