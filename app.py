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

from medical_ui_enhancements import apply_extended_ui, hero, model_guide

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

# FINAL ENSEMBLE: original + B3 alpha=0.50, both fine-tuned MobileNetV2.
FINAL_ENSEMBLE_FILES = {
    "original": "experiments/classification_ablation/original/best_finetuned.keras",
    "b3": "experiments/classification_ablation/b3_alpha050/best_finetuned.keras",
}

@st.cache_resource
def load_final_ensemble_models():
    models = {}
    for name, relative in FINAL_ENSEMBLE_FILES.items():
        path = BASE_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"Nihai ensemble modeli bulunamadı: {path}")
        models[name] = tf.keras.models.load_model(path, compile=False)
    return models


def analyze_final_ensemble(image, original_bytes):
    # Exact B3 pipeline from create_b3_softmask_datasets.py:
    # 256x256 grayscale -> U-Net V3 -> threshold -> refine -> nearest resize
    # -> outside-lung pixel intensity * 0.50 -> PNG -> TF bilinear 224x224.
    from create_b2_expanded_dataset import refine
    seg_model = load_segmentation_model()
    gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    x = np.asarray(gray, dtype=np.float32)[None, ..., None] / 255.0
    raw = seg_model.predict(x, verbose=0)[0, :, :, 0] >= 0.5
    refined = refine(raw)
    mask_img = Image.fromarray(refined.astype(np.uint8) * 255).resize(
        image.size, Image.Resampling.NEAREST
    )
    mask = np.asarray(mask_img) > 0
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    rgb[~mask] = np.rint(rgb[~mask].astype(np.float32) * 0.50).astype(np.uint8)
    b3_image = Image.fromarray(rgb)
    models = load_final_ensemble_models()
    original_batch = prepare_classification_image(original_bytes)
    b3_batch = prepare_classification_image(b3_image)
    p_original = np.asarray(models["original"].predict(original_batch, verbose=0)[0], dtype=np.float64)
    p_b3 = np.asarray(models["b3"].predict(b3_batch, verbose=0)[0], dtype=np.float64)
    if p_original.shape != (3,) or p_b3.shape != (3,):
        raise ValueError(f"Beklenmeyen sınıf çıktısı: {p_original.shape}, {p_b3.shape}")
    if not np.all(np.isfinite(p_original)) or not np.all(np.isfinite(p_b3)):
        raise ValueError("Model olasılıklarında geçersiz değer var")
    combined = 0.5 * p_original + 0.5 * p_b3
    idx = int(np.argmax(combined))
    return {
        "label": CLASS_NAMES[idx],
        "scores": {name: float(combined[i] * 100) for i, name in enumerate(CLASS_NAMES)},
        "original_scores": p_original.tolist(),
        "b3_scores": p_b3.tolist(),
        "b3_image": b3_image,
        "refined_mask": mask_img,
    }


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

# Page-local spacing; shared colors, sidebar and widgets come from the theme.
st.markdown(
    """<style>
    .pa-hero {padding:16px 20px;gap:16px;margin-bottom:12px}
    .pa-hero img {width:52px;height:52px}
    .pa-hero h1 {font-size:28px!important;margin:4px 0!important}
    .pa-hero .pa-badges:empty {display:none}
    [data-testid="stImage"] img {border-radius:12px}
    .st-key-xray_preview [data-testid="stImage"] img {
        max-height:260px;max-width:100%;width:auto!important;height:auto;object-fit:contain
    }
    </style>""",
    unsafe_allow_html=True,
)

hero("Görüntü Analizi", "X-Ray görüntünüzü yükleyin, analiz edin ve sonuçları inceleyin.",
     root=BASE_DIR)

st.warning(
    "Tıbbi kullanım uyarısı: Bu uygulama akademik araştırma prototipidir. "
    "Sonuçlar tıbbi tanı veya tedavi amacıyla kullanılamaz."
)

CLASS_LABELS = {
    "NORMAL": "Normal",
    "BACTERIAL": "Bakteriyel pnömoni",
    "VIRAL": "Viral pnömoni",
}

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

    with left, st.container(key="xray_preview"):

        st.markdown("### Akciğer Röntgeni Önizlemesi")

        st.image(

            image,

            # Keep source pixels intact; CSS limits only the displayed size.
            width=image.width,
            output_format="PNG"

        )

    with right:

        st.markdown("### Görüntü Analizi")

        if st.button(

            "Görüntüyü analiz et",

            type="primary",

            width="stretch"

        ):

            try:

                with st.spinner(

                    "Modeller görüntüyü analiz ediyor..."

                ):

                    final_ensemble = analyze_final_ensemble(image, file_bytes)

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

                    "final_ensemble": final_ensemble,
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

        st.write(
            "Analiz tamamlandığında nihai tahmini, akciğer maskesini ve "
            "araştırma amaçlı model karşılaştırmalarını aşağıdaki sekmelerde inceleyebilirsiniz."
        )
        st.caption("Görüntünüz bu oturumda tutulur; temizle butonuyla kaldırabilirsiniz.")

    analysis = st.session_state.get("analysis")

    if (

        analysis is not None

        and analysis["file_hash"] == file_hash
        and "final_ensemble" in analysis

    ):

        final_ensemble = analysis["final_ensemble"]
        results = analysis["results"]

        mask = analysis["mask"]

        ablation_results = analysis["ablation_results"]

        masked_image = analysis["masked_image"]

        tab_final, tab_seg, tab_cls, tab_abl = st.tabs([
            "Nihai Tahmin", "Segmentasyon", "Diğer Modeller", "Ablasyon Deneyleri"
        ])

        with tab_final:
            st.markdown("## Nihai tahmin")
            st.caption("MobileNetV2 ensemble · Orijinal + B3 alpha=0.50 · Eşit ağırlıklı skor ortalaması")
            with st.container(border=True):
                label = final_ensemble["label"]
                st.caption("TAHMİN EDİLEN SINIF")
                st.markdown(f"### {label} — {CLASS_LABELS[label]}")
                st.write(
                    f"Modelin en yüksek skor verdiği sınıf {label} ({CLASS_LABELS[label]}). "
                    "Aşağıdaki skorlar, modelin bu görüntü için üç sınıfa verdiği çıktıları "
                    "gösterir; kesin hastalık olasılığı veya klinik güven düzeyi değildir."
                )
                st.markdown("#### Sınıf skorları")
                for class_name, score in sorted(
                    final_ensemble["scores"].items(), key=lambda item: item[1], reverse=True
                ):
                    score_text = f"%{score:.2f}".replace(".", ",")
                    st.write(f"**{class_name} — {CLASS_LABELS[class_name]}** · {score_text}")
                    st.progress(min(1.0, max(0.0, score / 100.0)))

            st.markdown("### Analizde kullanılan görüntüler")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(image, caption="Orijinal X-Ray", width="stretch")
            with col_b:
                st.image(final_ensemble["b3_image"], caption="B3 · Akciğer dışı yoğunluk %50", width="stretch")
            st.download_button(
                "B3 görüntüsünü PNG indir",
                data=image_to_png_bytes(final_ensemble["b3_image"]),
                file_name="b3_alpha050.png", mime="image/png", key="final_b3_download",
            )
            with st.expander("Teknik ayrıntılar ve kayıtlı doğrulama sonucu", expanded=False):
                st.markdown(
                    "**B3 görüntü işleme:** Görüntü 256×256 gri tona çevrilir ve U-Net V3 "
                    "çıktısına 0,5 eşik uygulanır. Küçük bölgeler ve delikler düzeltilir, "
                    "maske genişletilir ve orijinal boyuta taşınır. Akciğer dışındaki piksel "
                    "yoğunlukları 0,50 ile çarpılır; bu işlem sert maskelemeden farklıdır. "
                    "Sınıflandırma girdileri 224×224 boyutuna dönüştürülür."
                )
                st.markdown(
                    "**Nihai yöntem:** Orijinal ve B3 görüntülerini kullanan iki MobileNetV2 "
                    "modelinin sınıf skorları %50 / %50 ağırlıkla birleştirilir. "
                    "Gösterilen yüzdeler softmax model skorlarıdır; kalibre edilmiş klinik olasılıklar değildir."
                )
                st.caption(
                    "Kayıtlı doğrulama: doğruluk %80,34 · makro F1 %80,49 (875 görüntü). "
                    "Bunlar doğrulama kümesinin toplu sonuçlarıdır; yüklenen görüntünün skorları değildir."
                )

        with tab_seg:
            st.markdown("## Akciğer segmentasyonu")
            st.write(
                "Segmentasyon, görüntüde akciğer olarak tahmin edilen bölgelerin piksel düzeyinde işaretlenmesidir. "
                "Yeşil katman bu bölgeleri X-Ray üzerinde gösterir; ikili maskede aynı bölgeler beyazdır."
            )
            opacity = st.slider(
                "Yeşil katmanın yoğunluğu (maske saydamlığı)",
                min_value=0.1, max_value=0.9, value=0.4, step=0.1,
                key="seg_overlay_opacity",
                help="Yalnızca yeşil katmanın görünümünü değiştirir; modelin ürettiği maske ve tahmin skorları değişmez.",
            )
            st.caption("Bu ayar yalnızca görsel overlay yoğunluğunu değiştirir; modelin ürettiği maskeyi değiştirmez.")
            show_overlay = st.checkbox(
                "Yeşil segmentasyon katmanını göster", value=True, key="seg_show_overlay",
            )
            overlay = make_overlay(image, mask, opacity)
            with st.container(border=True):
                seg_cols = st.columns(3)
                with seg_cols[0]:
                    st.image(image, caption="Orijinal X-Ray", width="stretch")
                with seg_cols[1]:
                    st.image(
                        overlay if show_overlay else image,
                        caption="Yeşil segmentasyon katmanı" if show_overlay else "Yeşil katman gizlendi",
                        width="stretch",
                    )
                with seg_cols[2]:
                    st.image(mask, caption="İkili akciğer maskesi", width="stretch", clamp=True)
            with st.container(horizontal=True):
                st.download_button(
                    "Akciğer maskesini PNG indir", key="pneumonia_download_718",
                    data=image_to_png_bytes(mask), file_name="lung_mask_v3.png", mime="image/png",
                )
                st.download_button(
                    "Yeşil katmanlı görseli PNG indir", key="pneumonia_download_725",
                    data=image_to_png_bytes(overlay), file_name="lung_overlay_v3.png", mime="image/png",
                )
            with st.expander("U-Net V3 mimarisi ve yöntem ayrıntıları", expanded=False):
                model_guide(["U-Net V3"], title="Segmentasyon modeli")
            st.warning(
                "Segmentasyon özellikle yoğun opasiteli pnömoni görüntülerinde eksik veya hatalı "
                "akciğer bölgeleri oluşturabilir. Bu maske doğrulanmış referans maske değildir."
            )

        with tab_cls:
            st.markdown("## Diğer modellerle karşılaştırma")
            st.write("Dört modelin aynı görüntü için ürettiği tahminleri ve sınıf skorlarını karşılaştırın.")
            st.dataframe(
                results.style.format({
                    class_name: lambda value: "%" + f"{value:.2f}".replace(".", ",")
                    for class_name in CLASS_NAMES
                }),
                width="stretch", hide_index=True,
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
                "Modellerin aynı tahmini yapması doğruluğu garanti etmez. "
                "Yüzdeler model skorlarıdır; kalibre edilmiş klinik olasılıklar değildir."
            )
            with st.expander("Dört modelin ayrıntılı kartları ve yöntemleri", expanded=False):
                preferred_model = "MobileNetV2 - Conservative Group"
                display_rows = results.to_dict("records")
                display_rows.sort(key=lambda row: row["Model"] != preferred_model)
                cols = st.columns(2)
                for index, row in enumerate(display_rows):
                    with cols[index % 2]:
                        with st.container(border=True):
                            st.markdown(f"### {row['Model']}")
                            st.metric("Model tahmini", row["Tahmin"])
                            for class_name in CLASS_NAMES:
                                score = row[class_name]
                                st.write(f"{class_name}: %{score:.2f}".replace(".", ","))
                                st.progress(min(1.0, max(0.0, score / 100)))
                model_guide(
                    ["Baseline CNN", "MobileNetV2", "Fine-Tuned MobileNetV2", "MobileNetV2 - Conservative Group"],
                    title="Modellerin çalışma ilkeleri",
                )
            csv_data = results.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Karşılaştırma sonuçlarını CSV indir", key="pneumonia_download_847",
                data=csv_data, file_name="model_comparison.csv", mime="text/csv",
            )

        with tab_abl:
            st.markdown("## Ablasyon deneyleri")
            st.write(
                "**Araştırma sorusu:** Akciğer dışındaki görüntü bilgisi tamamen kaldırıldığında "
                "sınıflandırma başarısı nasıl değişir? Deney A orijinal görüntüyü, "
                "Deney B ise akciğer dışı pikselleri sıfırlanmış (sert maskeli) görüntüyü kullanır."
            )
            st.markdown("### Kayıtlı doğrulama sonuçları")
            validation_cols = st.columns(2)
            with validation_cols[0]:
                st.metric("Deney A · Orijinal", "%79,89")
                st.caption("Doğrulama doğruluğu · Makro F1: %79,75")
            with validation_cols[1]:
                st.metric("Deney B · Sert maskeli", "%66,74")
                st.caption("Doğrulama doğruluğu · Makro F1: %66,75")
            st.write(
                "Deney B'nin doğrulama doğruluğu Deney A'dan **13,15 yüzde puan daha düşüktür**. "
                "Bu deneyde akciğer dışı bilgiyi tamamen kaldırmak, bu doğrulama kümesinde başarıyı azaltmıştır."
            )
            st.caption(
                "Bu değerler kayıtlı doğrulama kümesinin toplu sonuçlarıdır; "
                "yüklediğiniz görüntünün tahmin skorları değildir."
            )
            st.info(
                "Sert maskeleme deneyi, nihai B3 alpha=0.50 ensemble yöntemiyle aynı değildir. "
                "Deney B akciğer dışını sıfırlar. B3 dış bölgelerin yoğunluğunu %50'ye düşürür; "
                "nihai ensemble orijinal ve B3 modellerinin skorlarını eşit ağırlıkla birleştirir."
            )
            with st.container(border=True):
                img_cols = st.columns(2)
                with img_cols[0]:
                    st.image(image, caption="Deney A girdisi · Orijinal X-Ray", width="stretch")
                with img_cols[1]:
                    st.image(masked_image, caption="Deney B girdisi · Sert maskeli X-Ray", width="stretch")
            with st.expander("Deney yöntemleri ve yüklenen görüntünün model kartları", expanded=False):
                st.write(
                    "Her iki MobileNetV2 aynı başlangıç ağırlıkları ve eğitim ayarlarıyla eğitilmiştir. "
                    "Kesin hiperparametreler için eğitim konfigürasyonları esas alınmalıdır. "
                    "Deney B'nin sınıflandırma girdisine U-Net V3 ikili maskesi uygulanır. "
                    "Bu deney ile nihai ensemble arasındaki sonuç farkı yalnızca maske yoğunluğuna atfedilemez."
                )
                model_guide(["Deney A - Orijinal", "Deney B - Maskeli"], title="Deney yöntemleri")
                st.markdown("### Yüklenen görüntünün tahmin skorları")
                result_cols = st.columns(2)
                for col, row in zip(result_cols, ablation_results.to_dict("records")):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"### {row['Model']}")
                            st.metric("Model tahmini", row["Tahmin"])
                            for class_name in CLASS_NAMES:
                                score = row[class_name]
                                st.write(f"{class_name}: %{score:.2f}".replace(".", ","))
                                st.progress(min(1.0, max(0.0, score / 100.0)))
            st.markdown("### Yüklenen görüntü · Ablasyon karşılaştırma tablosu")
            st.caption("Bu tablodaki yüzdeler model skorlarıdır; klinik olasılık veya doğrulama doğruluğu değildir.")
            st.dataframe(
                ablation_results.style.format({
                    class_name: lambda value: "%" + f"{value:.2f}".replace(".", ",")
                    for class_name in CLASS_NAMES
                }),
                width="stretch", hide_index=True,
            )
            st.download_button(
                "Ablasyon sonuçlarını CSV indir", key="pneumonia_download_898",
                data=ablation_results.to_csv(index=False).encode("utf-8-sig"),
                file_name="ablation_comparison.csv", mime="text/csv",
            )
