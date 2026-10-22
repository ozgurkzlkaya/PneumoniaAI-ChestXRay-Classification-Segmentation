from pathlib import Path

# Ham veri setinin bulunduğu klasör
SOURCE = Path.home() / "Desktop" / "archive" / "chest_xray"

normal_count = 0
bacterial_count = 0
viral_count = 0
unknown_count = 0

normal_files = []
bacterial_files = []
viral_files = []
unknown_files = []

# train, test ve val içindeki tüm görüntüleri kontrol et
for split in ["train", "test", "val"]:

    # NORMAL
    normal_folder = SOURCE / split / "NORMAL"

    if normal_folder.exists():
        for file in normal_folder.iterdir():
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                normal_count += 1
                normal_files.append(file)

    # PNEUMONIA
    pneumonia_folder = SOURCE / split / "PNEUMONIA"

    if pneumonia_folder.exists():
        for file in pneumonia_folder.iterdir():

            if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            filename = file.name.lower()

            if "bacteria" in filename:
                bacterial_count += 1
                bacterial_files.append(file)

            elif "virus" in filename:
                viral_count += 1
                viral_files.append(file)

            else:
                unknown_count += 1
                unknown_files.append(file)


print("\n========== VERI SETI KONTROLU ==========\n")

print(f"NORMAL       : {normal_count}")
print(f"BACTERIAL    : {bacterial_count}")
print(f"VIRAL        : {viral_count}")
print(f"BILINMEYEN   : {unknown_count}")

total = normal_count + bacterial_count + viral_count + unknown_count

print("----------------------------------------")
print(f"TOPLAM       : {total}")
print("----------------------------------------")

if unknown_count > 0:
    print("\nSiniflandirilamayan dosyalar:")

    for file in unknown_files[:20]:
        print(file.name)

print("\nKontrol tamamlandi.")