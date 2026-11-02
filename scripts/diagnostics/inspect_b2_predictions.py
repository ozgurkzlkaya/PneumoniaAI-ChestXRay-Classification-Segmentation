from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent

a_path = root / "results/classification/b2_evaluation/predictions.csv"
b_path = root / "results/classification/b2_mask_sensitivity/predictions.csv"

a = pd.read_csv(a_path)
b = pd.read_csv(b_path)

print("=== EVALUATION CSV ===")
print("Columns:", list(a.columns))
print("Rows:", len(a))
print(a.head(3).to_string(index=False))

print("\n=== SENSITIVITY CSV ===")
print("Columns:", list(b.columns))
print("Rows:", len(b))
print(b.head(3).to_string(index=False))

b = b[b["iterations"] == 3].copy()

print("\n=== ITERATION 3 ===")
print("Rows:", len(b))
print("Classes:", b["true_class"].value_counts().to_dict())

print("\nNext: match predictions by class and filename.")
