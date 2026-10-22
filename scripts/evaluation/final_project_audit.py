
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import tensorflow as tf

BASE = Path(__file__).resolve().parents[2]

MODELS = {
    "Baseline CNN": "best_pneumonia_model.keras",
    "MobileNetV2": "best_transfer_model.keras",
    "Fine-Tuned MobileNetV2": "best_finetuned_model.keras",
    "U-Net V1": "best_segmentation_model.keras",
    "U-Net V2": "best_segmentation_model_v2.keras",
    "U-Net V3": "best_segmentation_model_v3.keras",
}

EXPECTED = {
    "Baseline CNN": (224, 224, 3, 3),
    "MobileNetV2": (224, 224, 3, 3),
    "Fine-Tuned MobileNetV2": (224, 224, 3, 3),
    "U-Net V1": (256, 256, 1, 1),
    "U-Net V2": (256, 256, 1, 1),
    "U-Net V3": (256, 256, 1, 1),
}

report = {
    "date": datetime.now().isoformat(),
    "tensorflow": tf.__version__,
    "models": {},
    "comparison_csv": {},
    "issues": [],
}

print("\n=== 1. MODEL DOSYALARI ===")

for name, filename in MODELS.items():
    path = BASE / filename
    item = {"file": filename, "exists": path.exists()}

    if not path.exists():
        print(f"[HATA] {name}: Dosya bulunamadi")
        report["issues"].append(f"{name}: model missing")
        report["models"][name] = item
        continue

    item["size_mb"] = round(path.stat().st_size / 1048576, 2)

    try:
        model = tf.keras.models.load_model(path, compile=False)

        input_shape = tuple(model.input_shape)
        output_shape = tuple(model.output_shape)

        item["input_shape"] = str(input_shape)
        item["output_shape"] = str(output_shape)
        item["parameters"] = model.count_params()

        h, w, c, out = EXPECTED[name]
        valid = (
            input_shape[1:] == (h, w, c)
            and output_shape[-1] == out
        )

        item["shape_valid"] = valid

        print(f"\n{name}")
        print(f"  Boyut: {item['size_mb']} MB")
        print(f"  Giris: {input_shape}")
        print(f"  Cikis: {output_shape}")
        print(f"  Parametre: {model.count_params():,}")
        print(f"  Durum: {'OK' if valid else 'KONTROL GEREKLI'}")

        if not valid:
            report["issues"].append(f"{name}: unexpected shape")

        del model

    except Exception as exc:
        item["load_error"] = str(exc)
        report["issues"].append(f"{name}: load failed")
        print(f"[HATA] {name}: {exc}")

    report["models"][name] = item


print("\n=== 2. SEGMENTASYON KARSILASTIRMA CSV ===")

csv_path = (
    BASE
    / "results" / "segmentation" / "segmentation_comparison_v1_v2_v3"
    / "comparison_results.csv"
)

if csv_path.exists():
    df = pd.read_csv(csv_path)

    report["comparison_csv"] = {
        "exists": True,
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
    }

    print(f"Satir sayisi: {len(df)}")
    print(f"Kolonlar: {list(df.columns)}")
    print(f"Eksik degerler: {df.isna().sum().sum()}")
    print("\nIlk 5 satir:")
    print(df.head().to_string(index=False))
else:
    report["comparison_csv"] = {"exists": False}
    report["issues"].append("Segmentation comparison CSV missing")
    print("[HATA] CSV bulunamadi")


print("\n=== 3. RAPOR ===")

output = BASE / "final_project_audit.json"

output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Rapor: {output}")
print(f"Bulunan sorun sayisi: {len(report['issues'])}")

if report["issues"]:
    for issue in report["issues"]:
        print(f" - {issue}")
else:
    print("Model dosyalari ve temel yapisal kontroller basarili.")
    print("Not: Bu test tahmin dogrulugunu olcmez.")
