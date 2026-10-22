
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "dataset"
REPORT = BASE / "final_classification_report"

CSV = REPORT / "advanced_leakage_matches.csv"
OUT = REPORT / "leakage_visual_review"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(CSV)

# Önce kesin kopyalar, sonra pHash yakın eşleşmeleri
priority = {
    "sha256": 0,
    "normalized_hash": 1,
    "phash": 2,
}

df["priority"] = df["type"].map(priority)
df = df.sort_values(
    ["priority", "distance"]
).reset_index(drop=True)

# Aynı dosya çiftinin tekrarlanan kayıtlarını kaldır
def pair_key(row):
    return tuple(sorted([
        str(row["path_a"]),
        str(row["path_b"]),
    ]))

df["pair_key"] = df.apply(pair_key, axis=1)
df = df.drop_duplicates("pair_key").reset_index(drop=True)

print(f"Tekil eslesme cifti: {len(df)}")

# Her sayfada 8 görüntü çifti
PAIRS_PER_PAGE = 8

for start in range(0, len(df), PAIRS_PER_PAGE):
    subset = df.iloc[start:start + PAIRS_PER_PAGE]

    fig, axes = plt.subplots(
        len(subset),
        2,
        figsize=(12, len(subset) * 3.2),
        squeeze=False,
    )

    for row_idx, (_, item) in enumerate(subset.iterrows()):
        for col_idx, side in enumerate(["a", "b"]):
            path = DATA / item[f"path_{side}"]

            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("L")
                axes[row_idx, col_idx].imshow(
                    img,
                    cmap="gray",
                )

            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(
                f"{item[f'split_{side}']} / "
                f"{item[f'class_{side}']}\n"
                f"{Path(item[f'path_{side}']).name}",
                fontsize=9,
            )

        axes[row_idx, 0].text(
            0.02,
            0.03,
            f"Eslesme: {item['type']} | "
            f"pHash mesafesi: {item['distance']}",
            transform=axes[row_idx, 0].transAxes,
            fontsize=8,
            color="yellow",
            bbox={
                "facecolor": "black",
                "alpha": 0.65,
                "pad": 4,
            },
        )

    fig.tight_layout()

    page = start // PAIRS_PER_PAGE + 1
    output = OUT / f"leakage_review_page_{page:02d}.png"

    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Kaydedildi:", output.name)

print("\nGorsel rapor tamamlandi.")
print("Klasor:", OUT)
