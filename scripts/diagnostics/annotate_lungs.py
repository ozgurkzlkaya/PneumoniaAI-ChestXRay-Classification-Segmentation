
from pathlib import Path
import csv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / "reference_segmentation"
IMAGES = ROOT / "images"
MASKS = ROOT / "masks"
MANIFEST = ROOT / "reference_manifest.csv"

SIZE = 512

with MANIFEST.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    rows = list(csv.DictReader(f))

MASKS.mkdir(parents=True, exist_ok=True)


class LungAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Akciger Referans Maske Cizimi")
        self.index = 0
        self.polygons = []
        self.points = []
        self.current_image = None
        self.original_size = None

        self.info = tk.Label(
            root, font=("Arial", 12, "bold")
        )
        self.info.pack(pady=8)

        self.canvas = tk.Canvas(
            root, width=SIZE, height=SIZE,
            bg="black"
        )
        self.canvas.pack()
        self.canvas.bind(
            "<Button-1>", self.add_point
        )

        controls = tk.Frame(root)
        controls.pack(pady=10)

        buttons = [
            ("Akcigeri Tamamla", self.finish_polygon),
            ("Son Noktayi Sil", self.undo),
            ("Maskeyi Kaydet", self.save),
            ("Onceki", self.previous),
            ("Sonraki", self.next_image),
        ]

        for label, command in buttons:
            tk.Button(
                controls, text=label,
                command=command
            ).pack(side="left", padx=4)

        self.status = tk.Label(
            root, font=("Arial", 10)
        )
        self.status.pack(pady=5)

        self.load_image()

    def image_path(self):
        row = rows[self.index]
        return (
            IMAGES / row["class"]
            / row["reference_filename"]
        )

    def mask_path(self):
        row = rows[self.index]
        return (
            MASKS / row["class"]
            / (Path(row["reference_filename"]).stem
               + ".png")
        )

    def load_image(self):
        path = self.image_path()

        with Image.open(path) as img:
            self.original_size = img.size
            self.current_image = img.convert(
                "RGB"
            ).resize((SIZE, SIZE))

        self.photo = ImageTk.PhotoImage(
            self.current_image
        )

        self.polygons = []
        self.points = []

        self.redraw()

        row = rows[self.index]
        self.info.config(
            text=(
                f"{self.index + 1}/{len(rows)} - "
                f"{row['class']} - "
                f"{row['reference_filename']}"
            )
        )

        self.update_status()

    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(
            0, 0, anchor="nw",
            image=self.photo
        )

        for polygon in self.polygons:
            coords = [
                value
                for point in polygon
                for value in point
            ]

            self.canvas.create_polygon(
                coords, outline="#00ff66",
                fill="", width=3
            )

        if self.points:
            coords = [
                value
                for point in self.points
                for value in point
            ]

            if len(self.points) >= 2:
                self.canvas.create_line(
                    coords, fill="yellow",
                    width=2
                )

            for x, y in self.points:
                self.canvas.create_oval(
                    x - 3, y - 3,
                    x + 3, y + 3,
                    fill="yellow"
                )

    def add_point(self, event):
        if len(self.polygons) >= 2:
            messagebox.showinfo(
                "Bilgi",
                "Iki akciger tamamlandi."
            )
            return

        self.points.append(
            (event.x, event.y)
        )
        self.redraw()

    def finish_polygon(self):
        if len(self.points) < 3:
            messagebox.showwarning(
                "Uyari",
                "En az 3 nokta secmelisin."
            )
            return

        self.polygons.append(
            self.points.copy()
        )
        self.points = []

        self.redraw()
        self.update_status()

    def undo(self):
        if self.points:
            self.points.pop()
        elif self.polygons:
            self.points = self.polygons.pop()

        self.redraw()
        self.update_status()

    def update_status(self):
        self.status.config(
            text=(
                f"Tamamlanan akciger: "
                f"{len(self.polygons)}/2 | "
                f"Aktif nokta: {len(self.points)}"
            )
        )

    def save(self):
        if len(self.polygons) != 2:
            messagebox.showwarning(
                "Eksik maske",
                "Iki akcigeri de tamamlamalisin."
            )
            return

        # Cizimi orijinal goruntu boyutuna tasir.
        width, height = self.original_size

        mask = Image.new(
            "L", (width, height), 0
        )
        draw = ImageDraw.Draw(mask)

        for polygon in self.polygons:
            scaled = [
                (
                    round(x * width / SIZE),
                    round(y * height / SIZE)
                )
                for x, y in polygon
            ]

            draw.polygon(
                scaled, fill=255
            )

        output = self.mask_path()
        output.parent.mkdir(
            parents=True, exist_ok=True
        )
        mask.save(output)

        messagebox.showinfo(
            "Kaydedildi",
            f"Maske kaydedildi:\n{output}"
        )

    def next_image(self):
        if self.index < len(rows) - 1:
            self.index += 1
            self.load_image()

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self.load_image()


app = tk.Tk()
LungAnnotator(app)
app.mainloop()
