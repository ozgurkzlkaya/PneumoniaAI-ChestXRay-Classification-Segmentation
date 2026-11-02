from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

root = Path(__file__).resolve().parent
data = root / "experiments/classification_ablation/b2_expanded/dataset/validation"

model_path = root / "experiments/classification_ablation/b2_expanded/best_finetuned.keras"

model = tf.keras.models.load_model(model_path, compile=False)

classes = ["NORMAL", "BACTERIAL", "VIRAL"]

for cls in classes:
    path = next(iter(sorted((data / cls).glob("*.png"))))

    pil_image = tf.keras.utils.load_img(
        path,
        target_size=(224, 224),
        color_mode="rgb"
    )
    x_pil = tf.keras.utils.img_to_array(pil_image)

    raw = tf.io.read_file(str(path))
    decoded = tf.io.decode_png(raw, channels=3)
    x_tf = tf.image.resize(
        decoded, (224, 224), method="bilinear"
    ).numpy()

    p_pil = model.predict(x_pil[None, ...], verbose=0)[0]
    p_tf = model.predict(x_tf[None, ...], verbose=0)[0]

    print()
    print("CLASS:", cls)
    print("IMAGE:", path.name)
    print("Mean pixel difference:", np.mean(np.abs(x_pil - x_tf)))
    print("Max pixel difference:", np.max(np.abs(x_pil - x_tf)))
    print("PIL prediction:", classes[int(np.argmax(p_pil))])
    print("TF prediction:", classes[int(np.argmax(p_tf))])
    print("PIL probabilities:", np.round(p_pil, 4))
    print("TF probabilities:", np.round(p_tf, 4))
