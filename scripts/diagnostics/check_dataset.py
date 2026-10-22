from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

DATASET = Path.home() / "Desktop" / "PneumoniaProject" / "dataset"

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(42)

# Her sınıftan 3 görüntü göstereceğiz
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

broken_files = []

for row, class_name in enumerate(CLASSES):

    folder = DATASET / "train" / class_name

    files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"{class_name}: {len(files)} train görüntüsü")

    selected = random.sample(files, 3)

    for col, file in enumerate(selected):

        try:
            img = Image.open(file).convert("L")

            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")

            axes[row, col].set_title(
                f"{class_name}\n{file.name[:30]}",
                fontsize=9
            )

        except Exception as e:

            broken_files.append(file)

            axes[row, col].axis("off")
            axes[row, col].set_title("HATA")


plt.suptitle(
    "Akciğer X-Ray Veri Seti Kontrolü",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "dataset_samples.png",
    dpi=200
)

plt.show()


print("\n==============================")
print("GORSEL KONTROL TAMAMLANDI")
print("==============================")

print("Bozuk görüntü sayısı:", len(broken_files))

if len(broken_files) == 0:
    print("Secilen orneklerde sorun bulunmadi.")
else:
    print("\nSorunlu dosyalar:")

    for file in broken_files:
        print(file)