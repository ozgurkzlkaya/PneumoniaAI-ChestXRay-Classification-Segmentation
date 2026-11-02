from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent

BASE = ROOT / "experiments/classification_ablation/b2_expanded"
MODEL = BASE / "best_finetuned.keras"

A = ROOT / "results/classification/b2_evaluation/predictions.csv"
B = ROOT / "results/classification/b2_mask_sensitivity/predictions.csv"

a = pd.read_csv(A)
b = pd.read_csv(B)
b = b[b["iterations"] == 3].copy()

a["filename"] = a["path"].map(lambda x: Path(x).stem)
b["filename"] = b["filename"].map(lambda x: Path(x).stem)

merged = a.merge(
    b,
    on=["true_class", "filename"],
    suffixes=("_saved", "_generated"),
    validate="one_to_one"
)

print("\n=== MATCHING ===")
print("Saved:", len(a))
print("Generated:", len(b))
print("Matched:", len(merged))

if len(merged) != 875:
    raise RuntimeError("Image matching incomplete!")

changed = merged[
    merged["predicted_class_saved"] !=
    merged["predicted_class_generated"]
].copy()

print("\n=== PREDICTION DIFFERENCES ===")
print("Changed predictions:", len(changed))

if len(changed):
    print(pd.crosstab(
        changed["predicted_class_saved"],
        changed["predicted_class_generated"]
    ).to_string())

    print("\nChanged by true class:")
    print(changed["true_class"].value_counts().to_string())

print("\n=== SAVED PNG WITH TF RESIZE ===")

model = tf.keras.models.load_model(MODEL, compile=False)
classes = ["NORMAL", "BACTERIAL", "VIRAL"]

correct = 0
predictions = []

for i, row in enumerate(a.itertuples(index=False), 1):
    raw = tf.io.read_file(row.path)
    image = tf.io.decode_png(raw, channels=3)
    image = tf.image.resize(
        image, (224, 224), method="bilinear"
    )

    scores = model.predict(image[None, ...], verbose=0)[0]
    predicted = classes[int(np.argmax(scores))]

    predictions.append(predicted)
    correct += int(predicted == row.true_class)

    if i % 100 == 0:
        print(f"{i}/875", flush=True)

print("\nSaved PNG + TensorFlow accuracy:",
      round(100 * correct / len(a), 3))

print("Saved PNG + PIL accuracy:",
      round(100 * (a["true_class"] ==
                   a["predicted_class"]).mean(), 3))

print("Generated masks + TensorFlow accuracy:",
      round(100 * (b["true_class"] ==
                   b["predicted_class"]).mean(), 3))

a["predicted_tf"] = predictions

print("\n=== PIL vs TF ON SAME SAVED PNG ===")
print("Changed predictions:",
      int((a["predicted_class"] != a["predicted_tf"]).sum()))

print("\n=== GENERATED vs SAVED TF ===")
comparison = a[["true_class", "filename", "predicted_tf"]].merge(
    b[["true_class", "filename", "predicted_class"]],
    on=["true_class", "filename"],
    validate="one_to_one"
)

print("Changed predictions:",
      int((comparison["predicted_tf"] !=
           comparison["predicted_class"]).sum()))

print("\nDiagnostic complete.")
