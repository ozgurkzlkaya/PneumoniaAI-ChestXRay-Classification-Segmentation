from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from medical_theme import apply_medical_theme, render_medical_sidebar
from medical_ui_enhancements import apply_extended_ui, hero, model_guide, intro  # PNEUMONIA_UI_V3

st.set_page_config(page_title="Pnömoni AI | Hata Analizi", page_icon="📊", layout="wide")
render_medical_sidebar()
apply_medical_theme()

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "experiments" / "classification_error_analysis" / "validation"
MODEL_FILES = {
    "Baseline CNN": "baseline_cnn",
    "MobileNetV2": "mobilenetv2",
    "Fine-Tuned MobileNetV2": "finetuned_mobilenetv2",
}
CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
SUMMARY_COLUMNS = {
    "model": "Model",
    "n": "Görüntü sayısı",
    "accuracy": "Doğruluk",
    "macro_f1": "Makro F1",
    "ece_10_bins": "ECE (10 aralık)",
    "multiclass_brier": "Çok sınıflı Brier skoru",
    "incorrect": "Hatalı tahmin",
    "high_confidence_wrong_90pct": "≥%90 skorlu hata",
}


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def safe_image_path(relative):
    candidate = (ROOT / str(relative).replace("\\", "/")).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def tr_number(value, digits=2):
    return f"{float(value):.{digits}f}".replace(".", ",")


def tr_percent(fraction):
    return "%" + tr_number(float(fraction) * 100)


def section_note(text):
    st.caption(text)


def show_metric_explanations():
    with st.expander("Performans metrikleri ne anlama geliyor?"):
        st.markdown(
            "**Doğruluk (Accuracy):** Tüm görüntüler içinde doğru sınıflandırılanların oranı.\n\n"
            "**Makro F1:** Her sınıfın F1 skorunun eşit ağırlıklı ortalaması; "
            "sınıf dengesizliğinde tek başına doğruluktan daha bilgilendirici olabilir.\n\n"
            "**Dengeli doğruluk:** Her sınıfın duyarlılığının (recall) ortalaması.\n\n"
            "**Precision (Kesinlik):** Bir sınıf olarak tahmin edilen örneklerin ne kadarı gerçekten o sınıftır?\n\n"
            "**Recall (Duyarlılık):** Gerçekte o sınıfa ait örneklerin ne kadarı doğru yakalanmıştır?\n\n"
            "**ROC-AUC:** Bir sınıfı diğerlerinden ayırt etme becerisinin eşikten bağımsız özeti.\n\n"
            "**ECE:** Tahmin skorları ile gözlenen doğruluk arasındaki kalibrasyon farkının özeti; "
            "daha düşük olması genellikle daha iyidir.\n\n"
            "**Brier skoru:** Tahmin edilen sınıf skorlarının gerçek etiketlerle uyuşmazlığını "
            "ölçer; daha düşük olması daha iyidir."
        )


apply_extended_ui()
hero("Hata Analizi", "Model değerlendirmesi | Yanlış tahminler, kalibrasyon ve test raporları",
     ["3 Model / 875 Doğrulama", "Ayrı 876 Test", "Kalibrasyon", "Hata İncelemesi"],
     root=Path(__file__).resolve().parents[1])
intro("<strong>Değerlendirme merkezi:</strong> Doğruluk, makro F1, yanlış tahminler "
      "ve kalibrasyon üzerinden modelleri inceleyin. Ayrı test ve doğrulama "
      "sonuçları doğrudan kıyaslanmamalıdır.")
st.info("Akademik araştırma prototipidir. Model skorları kalibre edilmiş klinik tanı olasılıkları değildir.")

if not REPORT_DIR.is_dir():
    st.error(f"Rapor klasörü bulunamadı: {REPORT_DIR}")
    st.stop()

model_name = st.selectbox("İncelenecek model", list(MODEL_FILES), key="error_analysis_model")
prefix = MODEL_FILES[model_name]
error_path = REPORT_DIR / f"{prefix}_errors.csv"
pred_path = REPORT_DIR / f"{prefix}_predictions.csv"
cm_path = REPORT_DIR / f"{prefix}_confusion_matrix.csv"
cal_path = REPORT_DIR / f"{prefix}_calibration.csv"

for path in [error_path, pred_path, cm_path, cal_path]:
    if not path.is_file():
        st.error(f"Eksik rapor: {path.name}")
        st.stop()

errors = load_csv(error_path)
cal = load_csv(cal_path)
# Predictions CSV is checked for completeness but not needed for the displayed reports.


# PNEUMONIA_BACTERIAL_ERROR_PANEL_V1
st.markdown("### BACTERIAL Sınıfı Hata İncelemesi")
st.caption(
    "Aynı 875 görüntülük doğrulama kümesindeki üç model "
    "karşılaştırılmaktadır. Conservative Group bu analize dahil değildir."
)

_bacterial_rows = []
_bacterial_predictions = {}

for _model_name, _prefix in MODEL_FILES.items():
    _prediction_file = REPORT_DIR / f"{_prefix}_predictions.csv"
    _df = load_csv(_prediction_file)

    _required = {"file", "actual", "predicted"}
    if not _required.issubset(_df.columns):
        st.error(f"Tahmin verilerinde gerekli sütunlar eksik: {_prediction_file.name}")
        st.stop()

    _df = _df.copy()
    _df["actual"] = _df["actual"].astype(str).str.upper().str.strip()
    _df["predicted"] = _df["predicted"].astype(str).str.upper().str.strip()

    _bacterial = _df[_df["actual"] == "BACTERIAL"]
    _total = len(_bacterial)
    _correct = int((_bacterial["predicted"] == "BACTERIAL").sum())
    _viral = int((_bacterial["predicted"] == "VIRAL").sum())
    _normal = int((_bacterial["predicted"] == "NORMAL").sum())

    _bacterial_rows.append({
        "Model": _model_name,
        "BACTERIAL sayısı": _total,
        "Doğru BACTERIAL": _correct,
        "VIRAL hatası": _viral,
        "NORMAL hatası": _normal,
        "BACTERIAL Recall (%)": round(
            100 * _correct / _total, 2
        ) if _total else 0.0,
    })
    _bacterial_predictions[_model_name] = _bacterial

_bacterial_summary = pd.DataFrame(_bacterial_rows)

st.dataframe(
    _bacterial_summary,
    hide_index=True,
    use_container_width=True,
)

st.markdown("#### BACTERIAL -> VIRAL hata karşılaştırması")
import altair as alt

_bacterial_chart_data = _bacterial_summary[
    ["Model", "VIRAL hatas\u0131"]
].copy()

_bacterial_chart = (
    alt.Chart(_bacterial_chart_data)
    .mark_bar(color="#087F8C", size=24)
    .encode(
        x=alt.X(
            "VIRAL hatas\u0131:Q",
            title="VIRAL olarak yanl\u0131\u015f s\u0131n\u0131fland\u0131r\u0131lan g\u00f6r\u00fcnt\u00fc",
        ),
        y=alt.Y(
            "Model:N",
            sort=None,
            title=None,
            axis=alt.Axis(labelLimit=300, labelFontSize=12),
        ),
        tooltip=[
            alt.Tooltip("Model:N", title="Model"),
            alt.Tooltip("VIRAL hatas\u0131:Q", title="Hata say\u0131s\u0131"),
        ],
    )
    .properties(height=165)
)

st.altair_chart(
    _bacterial_chart,
    use_container_width=True,
)


with st.expander("Yanlış VIRAL tahmin edilen X-Ray örneklerini incele"):
    _chosen_model = st.selectbox(
        "Model",
        list(MODEL_FILES),
        key="bacterial_error_inspection_model",
    )

    _mistakes = _bacterial_predictions[_chosen_model]
    _mistakes = _mistakes[
        _mistakes["predicted"] == "VIRAL"
    ].copy()

    st.write(f"Toplam BACTERIAL -> VIRAL hatası: {len(_mistakes)}")

    _columns = [
        col for col in [
            "file", "actual", "predicted", "confidence",
            "score_BACTERIAL", "score_VIRAL"
        ] if col in _mistakes.columns
    ]

    st.dataframe(
        _mistakes[_columns],
        hide_index=True,
        use_container_width=True,
    )

    if not _mistakes.empty:
        _selected_file = st.selectbox(
            "İncelenecek görüntü",
            _mistakes["file"].astype(str).tolist(),
            key="bacterial_error_inspection_file",
        )

        _image_path = safe_image_path(_selected_file)

        if _image_path is not None and Path(_image_path).is_file():
            st.image(
                str(_image_path),
                caption="Gerçek: BACTERIAL | Tahmin: VIRAL",
                width=480,
            )
        else:
            st.info("Görüntü dosyası bu konumda bulunamadı.")

st.info(
    "Bu bulgular veri kümesi etiketlerine göre hesaplanmıştır. "
    "Model skorları kalibre edilmiş klinik tanı olasılıkları değildir."
)
# END_PNEUMONIA_BACTERIAL_ERROR_PANEL_V1


# PNEUMONIA_SHARED_ERRORS_V1
with st.expander("Modellerin Ortak Hataları - 42 Görüntü"):
    _shared_frames = []

    for _name, _prefix in MODEL_FILES.items():
        _data = load_csv(
            REPORT_DIR / f"{_prefix}_predictions.csv"
        ).copy()

        _data["file"] = (
            _data["file"].astype(str)
            .str.replace("\\\\", "/", regex=False)
        )
        _data["actual"] = (
            _data["actual"].astype(str).str.upper().str.strip()
        )
        _data["predicted"] = (
            _data["predicted"].astype(str).str.upper().str.strip()
        )

        _data = _data[
            [
                "file", "actual", "predicted",
                "score_NORMAL", "score_BACTERIAL", "score_VIRAL"
            ]
        ].copy()

        _data = _data.rename(columns={
            "actual": f"actual_{_prefix}",
            "predicted": f"predicted_{_prefix}",
            "score_NORMAL": f"normal_{_prefix}",
            "score_BACTERIAL": f"bacterial_{_prefix}",
            "score_VIRAL": f"viral_{_prefix}",
        })

        _shared_frames.append(_data)

    _shared = _shared_frames[0]

    for _next in _shared_frames[1:]:
        _shared = _shared.merge(
            _next, on="file", how="inner", validate="one_to_one"
        )

    _prefixes = list(MODEL_FILES.values())
    _actual_cols = [f"actual_{p}" for p in _prefixes]

    if len(_shared) != 875:
        st.error("Doğrulama dosyalarındaki kayıtlar eşleşmiyor.")
        st.stop()

    if not all(
        (_shared[_actual_cols[0]] == _shared[col]).all()
        for col in _actual_cols[1:]
    ):
        st.error("Gerçek sınıf etiketleri eşleşmiyor.")
        st.stop()

    _common = _shared[
        (_shared[_actual_cols[0]] == "BACTERIAL")
        & (
            _shared[
                [f"predicted_{p}" for p in _prefixes]
            ] == "VIRAL"
        ).all(axis=1)
    ].copy()

    st.metric("Üç modelin ortak VIRAL hatası", len(_common))

    if not _common.empty:
        _selected = st.selectbox(
            "Ortak hata görüntüsü",
            _common["file"].tolist(),
            key="shared_error_image",
        )

        _row = _common[
            _common["file"] == _selected
        ].iloc[0]

        _image = safe_image_path(_selected)

        if _image is not None and Path(_image).is_file():
            st.image(
                str(_image),
                caption="Gerçek etiket: BACTERIAL",
                width=500,
            )
        else:
            st.warning("Görüntü dosyası bulunamadı.")

        _scores = []

        for _name, _prefix in MODEL_FILES.items():
            _scores.append({
                "Model": _name,
                "Tahmin": _row[f"predicted_{_prefix}"],
                "NORMAL (%)": round(
                    float(_row[f"normal_{_prefix}"]) * 100, 2
                ),
                "BACTERIAL (%)": round(
                    float(_row[f"bacterial_{_prefix}"]) * 100, 2
                ),
                "VIRAL (%)": round(
                    float(_row[f"viral_{_prefix}"]) * 100, 2
                ),
            })

        st.dataframe(
            pd.DataFrame(_scores),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Ortak hata listesini CSV indir",
            data=_common.to_csv(index=False).encode("utf-8-sig"),
            file_name="shared_bacterial_viral_errors.csv",
            mime="text/csv",
        )

    st.caption(
        "Ortak hata, modellerin aynı veri kümesi etiketine "
        "göre aynı yanlış sınıfı tahmin etmesidir. "
        "Bu sonuç klinik tanı veya görüntü bulgusu açıklaması değildir."
    )
# END_PNEUMONIA_SHARED_ERRORS_V1



model_guide(["Baseline CNN", "MobileNetV2", "Fine-Tuned MobileNetV2",
             "MobileNetV2 - Conservative Group"], title="İncelenen modeller hakkında")

st.markdown("### Model Değerlendirme Merkezi")
tab_overview, tab_errors, tab_cal, tab_test = st.tabs([
    "Genel Performans",
    "Hatalı Tahminler",
    "Kalibrasyon ve Karışıklık Matrisi",
    "Conservative Group Testi",
])

with tab_overview:
    st.subheader("Üç modelin doğrulama performansı")
    section_note(
        "Bu bölüm aynı doğrulama raporlarında yer alan üç modeli karşılaştırır. "
        "Conservative Group modelinin ayrı test kümesi bu tabloya dahil değildir."
    )
    summary_path = REPORT_DIR / "model_summary.csv"
    if summary_path.is_file():
        summary = load_csv(summary_path)
        available = [col for col in SUMMARY_COLUMNS if col in summary.columns]
        if available:
            if "accuracy" in summary.columns and "macro_f1" in summary.columns:
                best_acc = summary.loc[pd.to_numeric(summary["accuracy"], errors="coerce").idxmax()]
                best_f1 = summary.loc[pd.to_numeric(summary["macro_f1"], errors="coerce").idxmax()]
                m1, m2, m3 = st.columns(3)
                m1.metric("En yüksek doğruluk", tr_percent(best_acc["accuracy"]))
                m2.metric("En yüksek makro F1", tr_percent(best_f1["macro_f1"]))
                if "n" in summary.columns:
                    m3.metric("Doğrulama görüntüsü", str(int(best_acc["n"])))
                st.caption(
                    f"Doğruluk lideri: {best_acc['model']} | "
                    f"Makro F1 lideri: {best_f1['model']}"
                )
            display = summary[available].rename(columns=SUMMARY_COLUMNS).copy()
            formats = {}
            for original in ("accuracy", "macro_f1", "ece_10_bins", "multiclass_brier"):
                label = SUMMARY_COLUMNS[original]
                if label in display.columns:
                    formats[label] = (lambda x: tr_number(x, 4))
            st.dataframe(display.style.format(formats), hide_index=True, use_container_width=True)
        else:
            st.warning("Özet raporun sütunları beklenen biçimde değil.")
    else:
        st.warning("Model karşılaştırma raporu bulunamadı.")
    show_metric_explanations()
    st.caption("ECE ve Brier skorları olasılık kalibrasyonuyla ilişkilidir; yüksek doğruluk tek başına iyi kalibrasyon anlamına gelmez.")

with tab_errors:
    st.subheader("Hatalı tahminleri inceleme")
    section_note(f"Seçilen model: {model_name}. Filtreler yalnızca kayıtlı hatalı tahminlere uygulanır.")
    a, b, c = st.columns(3)
    with a:
        actual = st.selectbox("Gerçek sınıf", ["Tümü"] + CLASSES)
    with b:
        predicted = st.selectbox("Tahmin edilen sınıf", ["Tümü"] + CLASSES)
    with c:
        minimum_score = st.slider("En düşük yanlış tahmin skoru (%)", 0, 100, 0, 5)

    filtered = errors.copy()
    if actual != "Tümü":
        filtered = filtered[filtered["actual"] == actual]
    if predicted != "Tümü":
        filtered = filtered[filtered["predicted"] == predicted]
    filtered["confidence"] = pd.to_numeric(filtered["confidence"], errors="coerce")
    filtered = filtered[filtered["confidence"] >= minimum_score / 100]
    filtered = filtered.sort_values("confidence", ascending=False).reset_index(drop=True)

    x, y, z = st.columns(3)
    x.metric("Toplam hatalı tahmin", len(errors))
    y.metric("Filtreye uyan hata", len(filtered))
    high_score_wrong = int((pd.to_numeric(errors["confidence"], errors="coerce") >= 0.90).sum())
    z.metric("≥%90 skorlu hata", high_score_wrong)
    st.caption("≥%90 skorlu hata: Modelin yüksek softmax skoru verdiği hâlde yanlış sınıflandırdığı görüntü sayısıdır.")

    columns = [col for col in ["file", "actual", "predicted", "confidence", "score_NORMAL", "score_BACTERIAL", "score_VIRAL"] if col in filtered.columns]
    translated = {
        "file": "Dosya", "actual": "Gerçek sınıf", "predicted": "Tahmin edilen sınıf",
        "confidence": "En yüksek skor", "score_NORMAL": "Normal skoru",
        "score_BACTERIAL": "Bakteriyel skoru", "score_VIRAL": "Viral skoru",
    }
    display_errors = filtered[columns].rename(columns=translated)
    score_cols = [translated[col] for col in columns if col == "confidence" or col.startswith("score_")]
    st.dataframe(
        display_errors.style.format({col: lambda v: tr_percent(v) if pd.notna(v) else "-" for col in score_cols}),
        use_container_width=True, hide_index=True,
    )

    if len(filtered):
        st.subheader("Hatalı sınıflandırılan görüntü")
        index = st.number_input("İncelenecek satır (1'den başlar)", min_value=1, max_value=len(filtered), value=1, step=1)
        row = filtered.iloc[int(index) - 1]
        img_path = safe_image_path(row["file"])
        if img_path is None:
            st.warning(f"Görüntü dosyası bulunamadı: {row['file']}")
        else:
            left, right = st.columns([1, 1])
            with left:
                with Image.open(img_path) as image:
                    st.image(image.copy(), caption=str(row["file"]), use_container_width=True)
            with right:
                st.markdown(f"**Gerçek sınıf:** {row['actual']}")
                st.markdown(f"**Model tahmini:** {row['predicted']}")
                st.markdown(f"**En yüksek softmax skoru:** {tr_percent(row['confidence'])}")
                for cls in CLASSES:
                    key = f"score_{cls}"
                    if key in row and pd.notna(row[key]):
                        value = float(row[key])
                        st.write(f"{cls}: {tr_percent(value)}")
                        st.progress(max(0.0, min(1.0, value)))
    else:
        st.info("Seçilen filtrelere uygun hatalı tahmin bulunamadı.")

    st.download_button(
        "Filtrelenmiş hataları CSV olarak indir",
        filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{prefix}_filtered_errors.csv", mime="text/csv",
        key="error_analysis_filtered_download",
    )

with tab_cal:
    st.subheader("Karışıklık matrisi")
    section_note(f"Seçilen model: {model_name}. Satırlar gerçek sınıfı, sütunlar tahmin edilen sınıfı gösterir.")
    cm = load_csv(cm_path)
    if cm.shape[1] >= 4:
        cm = cm.iloc[:, :4].copy().set_index(cm.columns[0])
        cm.columns = CLASSES
    elif cm.shape[1] == 3:
        cm.index = CLASSES
        cm.columns = CLASSES
    else:
        st.warning("Karışıklık matrisi CSV biçimi beklenenden farklı.")
        cm = None
    if cm is not None:
        cm = cm.apply(pd.to_numeric, errors="coerce")
        st.dataframe(cm, use_container_width=True)
        if cm.shape == (3, 3) and cm.notna().all().all():
            values = cm.to_numpy(dtype=int)
            correct = int(np.trace(values))
            total = int(values.sum())
            m1, m2 = st.columns(2)
            m1.metric("Doğru tahmin", correct)
            m2.metric("Hatalı tahmin", total - correct)
            pairs = [(int(values[i, j]), CLASSES[i], CLASSES[j]) for i in range(3) for j in range(3) if i != j]
            count, real_cls, predicted_cls = max(pairs)
            st.info(f"En sık karıştırılan yön: Gerçekte {real_cls} olan {count} görüntü {predicted_cls} olarak tahmin edilmiş.")

    st.subheader("Kalibrasyon: model skoru ve gözlenen doğruluk")
    section_note("İyi kalibre edilmiş bir modelde, örneğin %80 skor verilen örneklerin yaklaşık %80'i doğru sınıflandırılır.")
    needed = ["bin_start", "bin_end", "count", "mean_confidence", "observed_accuracy"]
    if all(key in cal.columns for key in needed):
        cal = cal.copy()
        for key in needed:
            cal[key] = pd.to_numeric(cal[key], errors="coerce")
        cal = cal[(cal["count"] > 0) & cal["mean_confidence"].notna() & cal["observed_accuracy"].notna()]
        if len(cal):
            plot = cal[["mean_confidence", "observed_accuracy"]].rename(columns={
                "mean_confidence": "Ortalama model skoru",
                "observed_accuracy": "Gözlenen doğruluk",
            })
            plot.index = [f"%{a * 100:.0f}–%{b * 100:.0f}" for a, b in zip(cal["bin_start"], cal["bin_end"])]
            st.line_chart(plot)
            st.caption("Boş aralıklar gösterilmez. Az örnekli aralıklarda sonuçlar değişken olabilir; grafik klinik kalibrasyon kanıtı değildir.")
        else:
            st.warning("Kalibrasyon için dolu skor aralığı bulunamadı.")
    else:
        st.warning("Kalibrasyon raporunun sütunları beklenen biçimde değil.")
    st.caption("Bu bölüm kayıtlı doğrulama raporlarını okur; yeniden eğitim veya tahmin yapmaz.")

with tab_test:
    st.header("MobileNetV2 – Conservative Group | Test Analizi")
    st.caption(
        "Bu bölüm 876 görüntülük ayrı retrospektif test kümesine aittir. "
        "Diğer üç modelin doğrulama sonuçlarıyla doğrudan karşılaştırılmamalıdır."
    )
    conservative_dir = ROOT / "experiments" / "conservative_group_training"
    control_dir = conservative_dir / "control"
    figures_dir = conservative_dir / "figures"
    metrics_path = control_dir / "test_metrics.json"
    prob_path = control_dir / "test_probability_metrics.json"

    if metrics_path.is_file() and prob_path.is_file():
        with metrics_path.open(encoding="utf-8") as f:
            test_metrics = json.load(f)
        with prob_path.open(encoding="utf-8") as f:
            prob_metrics = json.load(f)

        st.subheader("Genel test performansı")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Doğruluk", tr_percent(test_metrics["accuracy"]))
        m2.metric("Makro F1", tr_percent(test_metrics["macro_f1"]))
        m3.metric("Dengeli doğruluk", tr_percent(test_metrics["balanced_accuracy"]))
        m4.metric("Makro ROC-AUC", tr_percent(prob_metrics["macro_roc_auc"]))
        st.caption(f"Test görüntüsü sayısı: {test_metrics['sample_count']}")
        show_metric_explanations()

        class_rows = []
        for class_name in CLASSES:
            values = test_metrics["per_class"][class_name]
            class_rows.append({
                "Sınıf": class_name,
                "Kesinlik (%)": values["precision"] * 100,
                "Duyarlılık (%)": values["recall"] * 100,
                "F1 (%)": values["f1"] * 100,
                "Görüntü sayısı": values["support"],
            })
        class_df = pd.DataFrame(class_rows)
        st.subheader("Sınıf bazında performans")
        st.dataframe(
            class_df.style.format({
                "Kesinlik (%)": tr_number,
                "Duyarlılık (%)": tr_number,
                "F1 (%)": tr_number,
            }),
            hide_index=True, use_container_width=True,
        )

        cm_values = np.asarray(test_metrics["confusion_matrix"], dtype=int)
        st.subheader("Test karışıklık matrisi")
        st.caption("Satırlar gerçek sınıfı, sütunlar tahmin edilen sınıfı gösterir.")
        st.dataframe(pd.DataFrame(cm_values, index=CLASSES, columns=CLASSES), use_container_width=True)
        correct = int(np.trace(cm_values))
        total = int(cm_values.sum())
        a, b = st.columns(2)
        a.metric("Doğru tahmin", correct)
        b.metric("Hatalı tahmin", total - correct)
        if cm_values.shape == (3, 3):
            confusions = [(int(cm_values[i, j]), CLASSES[i], CLASSES[j]) for i in range(3) for j in range(3) if i != j]
            count, real_cls, predicted_cls = max(confusions)
            st.info(
                f"En sık karışıklık: Gerçekte {real_cls} olan {count} görüntü "
                f"{predicted_cls} olarak sınıflandırılmış. Bu bulgu, sınıflar arasındaki "
                "ayrımın hangi yönde zorlaştığını gösterir."
            )

        st.subheader("ROC ve kesinlik–duyarlılık eğrileri")
        st.caption("ROC eğrisi ayırt etme gücünü; kesinlik–duyarlılık eğrisi ise iki metrik arasındaki dengeyi gösterir.")
        for filename, title in [
            ("roc_curves.png", "ROC eğrileri"),
            ("precision_recall_curves.png", "Kesinlik–duyarlılık eğrileri"),
            ("confusion_matrix.png", "Karışıklık matrisi görseli"),
        ]:
            figure_path = figures_dir / filename
            if figure_path.is_file():
                st.image(str(figure_path), caption=title, use_container_width=True)
            else:
                st.warning(f"Görsel bulunamadı: {filename}")
        st.caption(
            "Grafiklerin içindeki İngilizce yazılar, daha önce oluşturulmuş PNG dosyalarının "
            "parçasıdır. Bunları Türkçeleştirmek için grafik üretim kodu ayrıca güncellenmelidir."
        )
        st.caption(
            "Metrikler kayıtlı retrospektif test raporlarından okunur. "
            "Bu sayfa yeniden eğitim veya yeni model tahmini yapmaz."
        )
    else:
        st.warning("Conservative Group test metrik dosyaları bulunamadı.")
