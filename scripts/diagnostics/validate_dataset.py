from pathlib import Path
from PIL import Image
from collections import Counter

DATASET = Path.home() / "Desktop" / "PneumoniaProject" / "dataset"

SPLITS = ["train", "validation", "test"]
CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

broken_files = []
total = 0

print("\n========== TAM VERI SETI KONTROLU ==========\n")

for split in SPLITS:

    print(f"\n{split.upper()}")

    split_total = 0

    for class_name in CLASSES:

        folder = DATASET / split / class_name

        files = [
            f for f in folder.iterdir()
            if f.suffix.lower() in EXTENSIONS
        ]

        valid = 0

        for file in files:

            try:
                with Image.open(file) as img:
                    img.verify()

                valid += 1

            except Exception as e:

                broken_files.append((file, str(e)))

        print(
            f"{class_name:<12}: "
            f"{valid}/{len(files)} okunabilir"
        )

        split_total += len(files)

    print(f"TOPLAM      : {split_total}")

    total += split_total


print("\n============================================")
print("GENEL TOPLAM :", total)
print("BOZUK DOSYA  :", len(broken_files))
print("============================================")


if len(broken_files) == 0:

    print("\nBASARILI: Tum goruntuler okunabiliyor.")

else:

    print("\nSORUNLU DOSYALAR:")

    for file, error in broken_files:

        print(file)
        print(error)