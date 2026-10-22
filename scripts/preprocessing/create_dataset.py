from pathlib import Path
from collections import defaultdict, Counter
import shutil
import random
import re

# ============================================================
# AYARLAR
# ============================================================

SOURCE = Path.home() / "Desktop" / "archive" / "chest_xray"
OUTPUT = Path.home() / "Desktop" / "PneumoniaProject" / "dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(RANDOM_SEED)


# ============================================================
# HASTA ID
# ============================================================

def get_patient_id(filename):

    name = filename.lower()

    # person1000_bacteria_2931.jpeg gibi dosyalar
    match = re.search(r"(person\d+)", name)

    if match:
        return match.group(1)

    # NORMAL dosyalarının isimleri farklı olduğu için
    # dosya adını benzersiz kimlik olarak kullanıyoruz.
    return "normal_" + Path(filename).stem.lower()


# ============================================================
# TÜM GÖRÜNTÜLERİ TOPLA
# ============================================================

records = []

for original_split in ["train", "test", "val"]:

    normal_folder = SOURCE / original_split / "NORMAL"

    if normal_folder.exists():

        for file in normal_folder.iterdir():

            if file.suffix.lower() in IMAGE_EXTENSIONS:

                records.append(
                    (file, "NORMAL", get_patient_id(file.name))
                )


    pneumonia_folder = SOURCE / original_split / "PNEUMONIA"

    if pneumonia_folder.exists():

        for file in pneumonia_folder.iterdir():

            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            filename = file.name.lower()

            if "bacteria" in filename:

                records.append(
                    (file, "BACTERIAL", get_patient_id(file.name))
                )

            elif "virus" in filename:

                records.append(
                    (file, "VIRAL", get_patient_id(file.name))
                )


print("Toplam goruntu:", len(records))


# ============================================================
# HASTALARA GÖRE GRUPLA
# ============================================================

patients = defaultdict(list)

for file, class_name, patient_id in records:

    patients[patient_id].append(
        (file, class_name)
    )


patient_ids = list(patients.keys())

random.shuffle(patient_ids)

print("Toplam hasta/benzersiz kimlik:", len(patient_ids))


# ============================================================
# HASTALARI BÖL
# ============================================================

total_patients = len(patient_ids)

train_end = int(
    total_patients * TRAIN_RATIO
)

val_end = train_end + int(
    total_patients * VAL_RATIO
)


train_patients = set(
    patient_ids[:train_end]
)

validation_patients = set(
    patient_ids[train_end:val_end]
)

test_patients = set(
    patient_ids[val_end:]
)


# ============================================================
# ÇIKTI KLASÖRÜNÜ TEMİZLE
# ============================================================

if OUTPUT.exists():

    print("\nEski dataset klasoru siliniyor...")

    shutil.rmtree(OUTPUT)


classes = [
    "NORMAL",
    "BACTERIAL",
    "VIRAL"
]

for split in [
    "train",
    "validation",
    "test"
]:

    for class_name in classes:

        (
            OUTPUT /
            split /
            class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# DOSYALARI KOPYALA
# ============================================================

results = {
    "train": [],
    "validation": [],
    "test": []
}


print("\nGoruntuler kopyalaniyor...")


for patient_id, images in patients.items():

    if patient_id in train_patients:
        split = "train"

    elif patient_id in validation_patients:
        split = "validation"

    else:
        split = "test"


    for file, class_name in images:

        destination = (
            OUTPUT /
            split /
            class_name /
            file.name
        )

        shutil.copy2(
            file,
            destination
        )

        results[split].append(
            (file, class_name, patient_id)
        )


# ============================================================
# SONUÇLAR
# ============================================================

print("\n========== VERI SETI DAGILIMI ==========")

grand_total = 0

for split in [
    "train",
    "validation",
    "test"
]:

    counter = Counter(
        class_name
        for _, class_name, _ in results[split]
    )

    print(f"\n{split.upper()}")

    split_total = 0

    for class_name in classes:

        count = counter[class_name]

        print(
            f"{class_name:<12}: {count}"
        )

        split_total += count

    print(
        f"{'TOPLAM':<12}: {split_total}"
    )

    grand_total += split_total


print("\n----------------------------------")
print("GENEL TOPLAM :", grand_total)
print("----------------------------------")


# ============================================================
# HASTA SIZINTISI KONTROLÜ
# ============================================================

train_ids = {
    patient
    for _, _, patient in results["train"]
}

validation_ids = {
    patient
    for _, _, patient in results["validation"]
}

test_ids = {
    patient
    for _, _, patient in results["test"]
}


train_val_overlap = (
    train_ids & validation_ids
)

train_test_overlap = (
    train_ids & test_ids
)

val_test_overlap = (
    validation_ids & test_ids
)


print(
    "\n========== HASTA SIZINTISI KONTROLU =========="
)

print(
    "\nTrain - Validation ortak hasta:",
    len(train_val_overlap)
)

print(
    "Train - Test ortak hasta:",
    len(train_test_overlap)
)

print(
    "Validation - Test ortak hasta:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):

    print(
        "\nBASARILI: Hasta sizintisi bulunmadi."
    )

else:

    print(
        "\nUYARI: Hasta sizintisi bulundu!"
    )


print("\nVeri seti hazir:")
print(OUTPUT)