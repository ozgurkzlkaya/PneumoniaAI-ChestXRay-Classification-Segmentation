# PNEUMONIA_CLASSIFICATION_LAYOUT_FIX_V2
# PNEUMONIA_DOWNLOAD_KEYS_V1
# PNEUMONIA_ABLATION_RESULT_CARDS_V2
# PNEUMONIA_SEG_ABL_PANELS_V2

from pathlib import Path
from io import BytesIO
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


# PNEUMONIA_AI_VISUAL_REFRESH_V1
# Unified theme and sidebar are defined in medical_theme.py
from medical_theme import apply_medical_theme, render_medical_sidebar
from medical_ui_enhancements import apply_extended_ui, hero, model_guide, intro  # PNEUMONIA_UI_V3

st.set_page_config(
    page_title="Pneumonia AI | Görüntü Analizi",
    page_icon="🫁",
    layout="wide"
)

apply_medical_theme()
render_medical_sidebar()
apply_extended_ui()

BASE_DIR = Path(__file__).resolve().parent

CLASS_NAMES = ["NORMAL", "BACTERIAL", "VIRAL"]

MODEL_FILES = {
    "Baseline CNN": "best_pneumonia_model.keras",
    "MobileNetV2": "best_transfer_model.keras",
    "Fine-Tuned MobileNetV2": "best_finetuned_model.keras",
    "MobileNetV2 - Conservative Group": "experiments/conservative_group_training/control/best_finetuned.keras"
}

SEGMENTATION_MODEL_FILE = "best_segmentation_model_v3.keras"

ABLATION_MODEL_FILES = {
    "Deney A - Orijinal": "experiments/classification_ablation/original/best_finetuned.keras",
    "Deney B - Maskeli": "experiments/classification_ablation/masked/best_finetuned.keras",
}


@st.cache_resource
def load_classification_models():
    models = {}

    for name, filename in MODEL_FILES.items():
        path = BASE_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Sınıflandırma modeli bulunamadı: {path}"
            )

        models[name] = tf.keras.models.load_model(
            path, compile=False
        )

    return models


@st.cache_resource
def load_ablation_models():
    models = {}
    for name, filename in ABLATION_MODEL_FILES.items():
        path = BASE_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Deney modeli bulunamadı: {path}")
        models[name] = tf.keras.models.load_model(path, compile=False)
    return models


@st.cache_resource
def load_segmentation_model():
    path = BASE_DIR / SEGMENTATION_MODEL_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Segmentasyon modeli bulunamadı: {path}"
        )

    return tf.keras.models.load_model(
        path, compile=False
    )


def prepare_classification_image(image):
    if isinstance(image, bytes):
        image_bytes = tf.convert_to_tensor(image)
        decoded = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )
        decoded.set_shape((None, None, 3))
    else:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        decoded = tf.io.decode_png(
            tf.convert_to_tensor(buffer.getvalue()),
            channels=3
        )

    resized = tf.image.resize(
        decoded,
        (224, 224),
        method="bilinear"
    )

    return tf.expand_dims(resized, axis=0).numpy()



def prepare_segmentation_image(image):
    # V3 modelinin eğitim ön işlemi:
    # 256x256, grayscale, [0,1]
    gray = image.convert("L")
    resized = gray.resize(
        (256, 256), Image.Resampling.BILINEAR
    )

    array = np.asarray(
        resized, dtype=np.float32
    ) / 255.0

    return array[np.newaxis, ..., np.newaxis]


def analyze_classification(image, original_bytes=None):
    models = load_classification_models()
    batch = prepare_classification_image(original_bytes if original_bytes is not None else image)

    results = []

    for model_name, model in models.items():
        scores = model.predict(
            batch, verbose=0
        )[0]

        predicted_index = int(np.argmax(scores))

        results.append({
            "Model": model_name,
            "Tahmin": CLASS_NAMES[predicted_index],
            "NORMAL": float(scores[0] * 100),
            "BACTERIAL": float(scores[1] * 100),
            "VIRAL": float(scores[2] * 100)
        })

    return pd.DataFrame(results)


def analyze_segmentation(image):
    model = load_segmentation_model()
    batch = prepare_segmentation_image(image)

    prediction = model.predict(
        batch, verbose=0
    )[0, :, :, 0]

    # Eğitim ve testte kullanılan eşik
    binary_mask = (prediction >= 0.5).astype(
        np.uint8
    ) * 255

    # Maskeyi orijinal görüntü boyutuna taşı
    mask_image = Image.fromarray(binary_mask)
    mask_image = mask_image.resize(
        image.size, Image.Resampling.NEAREST
    )

    return mask_image


def create_masked_classification_image(image, mask):
    # Ablasyon veri setindeki gibi: orijinal boyutta maskele,
    # akciğer dışını sıfırla, ardından 224x224 boyutuna getir.
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    binary_mask = np.asarray(mask.convert("L")) > 0
    masked = np.where(binary_mask[..., None], original, 0).astype(np.uint8)
    return Image.fromarray(masked, mode="RGB")


def analyze_ablation(image, mask, original_bytes=None):
    models = load_ablation_models()
    masked_image = create_masked_classification_image(image, mask)

    input_images = {
        "Deney A - Orijinal": (
            original_bytes if original_bytes is not None else image
        ),
        "Deney B - Maskeli": masked_image,
    }

    results = []
    for model_name, model in models.items():
        batch = prepare_classification_image(input_images[model_name])
        scores = model.predict(batch, verbose=0)[0]
        predicted_index = int(np.argmax(scores))
        results.append({
            "Model": model_name,
            "Tahmin": CLASS_NAMES[predicted_index],
            "NORMAL": float(scores[0] * 100),
            "BACTERIAL": float(scores[1] * 100),
            "VIRAL": float(scores[2] * 100),
        })

    return pd.DataFrame(results), masked_image


def make_overlay(image, mask, opacity=0.4):
    original = image.convert("RGB")

    mask_array = np.asarray(mask) > 0
    original_array = np.asarray(
        original, dtype=np.float32
    ).copy()

    # Yeşil akciğer maskesi
    green = np.array(
        [40, 220, 75], dtype=np.float32
    )

    original_array[mask_array] = (
        (1.0 - opacity) * original_array[mask_array]
        + opacity * green
    )

    return Image.fromarray(
        np.clip(original_array, 0, 255).astype(
            np.uint8
        )
    )


def image_to_png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()




st.markdown(
    """
    <style>
    .stApp {
        background: #F5FAFA;
        color: #173B46;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2ECEB;
    }

    [data-testid="stSidebarNav"] a {
        border-radius: 9px;
        margin: 4px 8px;
    }

    [data-testid="stSidebarNav"] a:hover {
        background: #E8F7F5;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: #173B46 !important;
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 750 !important;
    }

    h2 {
        font-weight: 680 !important;
    }

    h3 {
        font-weight: 650 !important;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px solid #E0EAE9;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #F8FCFB;
        border: 1px dashed #A9D7D3;
        border-radius: 10px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2ECEB;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #E2ECEB;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E2ECEB;
        border-radius: 12px;
        overflow: hidden;
    }

    div.stButton > button[kind="primary"] {
        background: #0E938F;
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #087D79;
        color: #FFFFFF;
    }

    div.stButton > button[kind="secondary"] {
        background: #FFFFFF;
        border: 1px solid #D6E6E4;
        border-radius: 10px;
        color: #173B46;
    }

    div[data-baseweb="tab-list"] {
        background: #FFFFFF;
        border: 1px solid #E2ECEB;
        border-radius: 12px;
        padding: 6px;
        gap: 6px;
    }

    button[data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #E4F5F2;
        color: #087D79;
    }

    [data-testid="stAlert"] {
        border-radius: 11px;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    .med-hero {
        background: #FFFFFF;
        border: 1px solid #E0EAE9;
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 20px;
        box-shadow: 0 5px 24px rgba(25, 70, 78, 0.035);
    }

    .med-eyebrow {
        color: #0E938F;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.13em;
        margin-bottom: 10px;
    }

    .med-title {
        color: #173B46;
        font-size: 33px;
        line-height: 1.15;
        font-weight: 750;
        letter-spacing: -0.04em;
    }

    .med-subtitle {
        color: #637D82;
        font-size: 14px;
        margin: 10px 0 16px 0;
    }

    .med-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .med-pill {
        background: #EFF8F7;
        color: #357A77;
        border: 1px solid #D7EBE8;
        border-radius: 999px;
        padding: 6px 11px;
        font-size: 11px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True
)

hero("Pneumonia AI", "Akciğer X-Ray Analizi | Çok Sınıflı Sınıflandırma ve Segmentasyon",
     ["4 Ana Model", "2 Ablasyon Deneyi", "U-Net V3", "Akademik Prototip"],
     root=Path(__file__).resolve().parent)
intro("<strong>Görüntü analizi:</strong> X-Ray dosyanızı yükleyerek dört sınıflandırıcıyı, "
      "akciğer segmentasyonunu ve orijinal/maskeli görüntü deneylerini inceleyebilirsiniz. "
      "Skorlar klinik tanı olasılıkları değildir.")

st.info(
    "Akademik araştırma prototipidir. "
    "Sonuçlar tıbbi tanı veya tedavi amacıyla kullanılamaz."
)

if "xray_uploader_version" not in st.session_state:
    st.session_state["xray_uploader_version"] = 0

uploaded_file = st.file_uploader(
    "Akci\u011fer X-Ray g\u00f6r\u00fcnt\u00fcs\u00fc y\u00fckleyin",
    type=["jpg", "jpeg", "png"],
    key=f"xray_uploader_{st.session_state['xray_uploader_version']}"
)

if uploaded_file is not None:
    st.session_state["saved_xray_bytes"] = uploaded_file.getvalue()
    st.session_state["saved_xray_name"] = uploaded_file.name

saved_bytes = st.session_state.get("saved_xray_bytes")

if saved_bytes is not None:
    if uploaded_file is None:
        st.info(
            "\u00d6nceki X-Ray g\u00f6r\u00fcnt\u00fcs\u00fc oturumdan geri y\u00fcklendi."
        )

    if st.button("Kay\u0131tl\u0131 g\u00f6r\u00fcnt\u00fcy\u00fc temizle"):
        st.session_state.pop("saved_xray_bytes", None)
        st.session_state.pop("saved_xray_name", None)
        st.session_state.pop("analysis", None)
        st.session_state["xray_uploader_version"] += 1
        st.rerun()

    file_bytes = saved_bytes
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    try:
        image = Image.open(
            BytesIO(file_bytes)
        ).convert("RGB")
    except Exception as exc:
        st.error(f"Görüntü açılamadı: {exc}")
        st.stop()

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("### Akciğer Röntgeni Önizlemesi")
        st.image(
            image,
            use_container_width=True
        )

    with right:
        st.markdown("### Görüntü Analizi")

        if st.button(
            "Sınıflandırma ve Segmentasyon Yap",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner(
                    "Modeller görüntüyü analiz ediyor..."
                ):
                    results = analyze_classification(
                        image,
                        original_bytes=file_bytes
                    )

                    mask = analyze_segmentation(
                        image
                    )
                    ablation_results, masked_image = analyze_ablation(
                        image,
                        mask,
                        original_bytes=file_bytes
                    )

                st.session_state["analysis"] = {
                    "file_hash": file_hash,
                    "results": results,
                    "mask": mask,
                    "ablation_results": ablation_results,
                    "masked_image": masked_image
                }

            except Exception as exc:
                st.error(
                    f"Analiz sırasında hata oluştu: {exc}"
                )


        st.caption(
            "4 sınıflandırma modeli | 2 ablasyon modeli | U-Net V3"
        )
        st.caption(
            "Yalnızca araştırma prototipidir. Klinik tanı amacıyla kullanılamaz."
        )

    analysis = st.session_state.get("analysis")

    if (
        analysis is not None
        and analysis["file_hash"] == file_hash
    ):
        results = analysis["results"]
        mask = analysis["mask"]
        ablation_results = analysis["ablation_results"]
        masked_image = analysis["masked_image"]

        tab_seg, tab_cls, tab_abl = st.tabs([
            "Akciğer Segmentasyonu",
            "Model Karşılaştırması",
            "Ablasyon Deneyleri"
        ])

        with tab_seg:
            st.divider()
            st.markdown("## 1. Akciğer Segmentasyonu")

            model_guide(["U-Net V3"], title="Segmentasyon mimarisi ve çalışma mantığı")

            opacity = st.slider(
                "Maske saydamlığı",
                min_value=0.1,
                max_value=0.9,
                value=0.4,
                step=0.1
            )

            show_overlay = st.checkbox(
                "Yeşil segmentasyon katmanını göster",
                value=True
            )

            overlay = make_overlay(
                image, mask, opacity
            )

            st.caption('Üç görüntü birlikte incelenir: orijinal X-Ray, segmentasyon görselleştirmesi ve ikili akciğer maskesi. Üretilen maske gerçek anotasyon değildir.')
            with st.container(border=True):
                seg_cols = st.columns(3)

                with seg_cols[0]:
                    st.image(
                        image,
                        caption="Orijinal görüntü",
                        use_container_width=True
                    )

                with seg_cols[1]:
                    st.image(
                        overlay if show_overlay else image,
                        caption=(
                            "U-Net V3 segmentasyonu"
                            if show_overlay
                            else "Maske gizlendi"
                        ),
                        use_container_width=True
                    )

                with seg_cols[2]:
                    st.image(
                        mask,
                        caption="İkili akciğer maskesi",
                        use_container_width=True,
                        clamp=True
                    )

            st.download_button(
                "Segmentasyon Maskesini PNG İndir",
                     key="pneumonia_download_718",
                data=image_to_png_bytes(mask),
                file_name="lung_mask_v3.png",
                mime="image/png"
            )

            st.download_button(
                "Segmentasyon Görselini PNG İndir",
                     key="pneumonia_download_725",
                data=image_to_png_bytes(overlay),
                file_name="lung_overlay_v3.png",
                mime="image/png"
            )

            st.warning(
                "Segmentasyon özellikle yoğun opasiteli "
                "pnömoni görüntülerinde eksik veya hatalı "
                "akciğer bölgeleri oluşturabilir. "
                "Bu maske doğrulanmış referans maske değildir."
            )


        with tab_cls:
            st.divider()
            st.markdown("## 2. Mevcut Modellerle Pnömoni Sınıflandırması")

            model_guide(["Baseline CNN", "MobileNetV2", "Fine-Tuned MobileNetV2",
                         "MobileNetV2 - Conservative Group"],
                        title="Sınıflandırma modellerini tanıyın")

            preferred_model = "MobileNetV2 - Conservative Group"

            display_rows = results.to_dict("records")
            display_rows.sort(
                key=lambda row: row["Model"] != preferred_model
            )

            cols = st.columns(2)

            for index, row in enumerate(display_rows):
                col = cols[index % 2]
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f"### {row['Model']}"
                        )

                        st.metric(
                            "Model Tahmini",
                            row["Tahmin"]
                        )

                        for class_name in CLASS_NAMES:
                            score = row[class_name]

                            st.write(
                                f"{class_name}: %{score:.2f}".replace(".", ",")
                            )

                            st.progress(
                                min(
                                    1.0,
                                    max(0.0, score / 100)
                                )
                            )

            # PNEUMONIA_COMPARISON_PANEL_V1
            st.info("Her kart aynı X-Ray görüntüsünün farklı bir model tarafından sınıflandırılmasını gösterir. NORMAL, BACTERIAL ve VIRAL değerleri softmax skorlarıdır; klinik tanı olasılığı değildir.")

            with st.container(border=True):
                st.markdown("### Model Karşılaştırma Tablosu")

            st.dataframe(
                results.style.format({
                    "NORMAL": lambda value: "%" + f"{value:.2f}".replace(".", ","),
                    "BACTERIAL": lambda value: "%" + f"{value:.2f}".replace(".", ","),
                    "VIRAL": lambda value: "%" + f"{value:.2f}".replace(".", ",")
                }),
                use_container_width=True,
                hide_index=True
            )



            predictions = results["Tahmin"].tolist()
            model_count = len(predictions)

            if model_count:
                from collections import Counter

                counts = Counter(predictions)
                top_class, top_count = counts.most_common(1)[0]

                if top_count == model_count:
                    st.success(
                        f"{model_count} modelin tamam\u0131 "
                        f"{top_class} s\u0131n\u0131f\u0131n\u0131 tahmin etti."
                    )
                elif top_count > model_count / 2:
                    different = results.loc[
                        results["Tahmin"] != top_class,
                        ["Model", "Tahmin"]
                    ]
                    details = "; ".join(
                        f"{row.Model} -> {row.Tahmin}"
                        for row in different.itertuples(index=False)
                    )
                    st.warning(
                        f"{model_count} modelden {top_count}'\u00fc "
                        f"{top_class} tahmininde birle\u015fiyor. "
                        f"Farkl\u0131 tahmin: {details}."
                    )
                else:
                    distribution = ", ".join(
                        f"{count} {label}"
                        for label, count in counts.most_common()
                    )
                    st.warning(
                        "Modeller aras\u0131nda \u00e7o\u011funluk olu\u015fmad\u0131. "
                        f"Tahmin da\u011f\u0131l\u0131m\u0131: {distribution}."
                    )

            st.caption(
                "Gösterilen yüzdeler softmax model "
                "skorlarıdır; kalibre edilmiş klinik "
                "olasılıklar değildir. Modellerin aynı "
                "fikirde olması doğruluğu garanti etmez."
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "Sınıflandırma Sonuçlarını CSV İndir",
                     key="pneumonia_download_847",
                data=csv_data,
                file_name="model_comparison.csv",
                mime="text/csv"
            )


        with tab_abl:
            st.divider()
            st.markdown("## 3. Ablasyon Deneyi: Orijinal ve Maskeli Akciğer Röntgeni")
            st.caption(
                "Her iki MobileNetV2 aynı başlangıç ağırlıkları ve eğitim ayarlarıyla "
                "eğitildi. Deney B'de sınıflandırma girdisine U-Net V3 maskesi uygulanır."
            )

            st.caption('Deney A orijinal X-Ray ile, Deney B ise segmentasyon maskesi uygulanmış X-Ray ile çalışır.')
            with st.container(border=True):
                img_cols = st.columns(2)
                with img_cols[0]:
                    st.image(image, caption="Deney A girdisi: orijinal X-Ray", use_container_width=True)
                with img_cols[1]:
                    st.image(masked_image, caption="Deney B girdisi: maskelenmiş X-Ray", use_container_width=True)

            model_guide(["Deney A - Orijinal", "Deney B - Maskeli"],
                        title="Ablasyon deneylerinin ayrıntıları")

            result_cols = st.columns(2)
            for col, row in zip(result_cols, ablation_results.to_dict("records")):
                with col:
                    with st.container(border=True):
                        st.markdown(f"### {row['Model']}")
                        st.metric("Model Tahmini", row["Tahmin"])
                        for class_name in CLASS_NAMES:
                            score = row[class_name]
                            st.write(f"{class_name}: %{score:.2f}".replace(".", ","))
                            st.progress(min(1.0, max(0.0, score / 100.0)))

            st.markdown("### Ablasyon Model Karşılaştırma Tablosu")
            st.dataframe(
                ablation_results.style.format({
                    "NORMAL": lambda value: "%" + f"{value:.2f}".replace(".", ","),
                    "BACTERIAL": lambda value: "%" + f"{value:.2f}".replace(".", ","),
                    "VIRAL": lambda value: "%" + f"{value:.2f}".replace(".", ","),
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Doğrulama kümesinde Deney A: %79,89 doğruluk / %79,75 makro F1; "
                "Deney B: %66,74 doğruluk / %66,75 makro F1. "
                "Bu değerler yüklenen görüntüye ait güven puanları değildir. "
                "Tahmin skorları kalibre edilmiş klinik olasılıklar değildir."
            )
            st.download_button(
                "Ablasyon Sonuçlarını CSV İndir",
                     key="pneumonia_download_898",
                data=ablation_results.to_csv(index=False).encode("utf-8-sig"),
                file_name="ablation_comparison.csv",
                mime="text/csv",
            )

