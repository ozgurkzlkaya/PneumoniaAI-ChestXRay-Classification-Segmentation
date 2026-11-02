from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from medical_theme import apply_medical_theme, render_medical_sidebar

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "assets" / "pneumonia_ai_logo.png"
st.set_page_config(page_title="Pnömoni AI | Hata Analizi", page_icon=str(LOGO) if LOGO.exists() else "🫁", layout="wide")
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



# Nihai değerlendirme kayıtları: bu sayfada model yeniden çalıştırılmaz.
ENSEMBLE_VALIDATION_CANDIDATES = [
    ROOT / "results/classification/ablation_uniform_evaluation/ensemble_50_50/validation_predictions.csv",
    ROOT / "results/classification/ablation_uniform_evaluation/ensemble_50_50/predictions.csv",
    ROOT / "results/classification/ablation_uniform_evaluation/ensemble_50_50/ensemble_predictions.csv",
    ROOT / "results/classification/ablation_uniform_evaluation/ensemble_50_50_predictions.csv",
    ROOT / "results/classification/ensemble_predictions.csv",
]
ENSEMBLE_TEST_PREDICTIONS = ROOT / "results/classification/fixed_ensemble_test_evaluation/test_predictions.csv"
ENSEMBLE_TEST_METRICS = ROOT / "results/classification/fixed_ensemble_test_evaluation/metrics.json"
ENSEMBLE_SUMMARY = ROOT / "results/classification/ablation_uniform_evaluation/ensemble_50_50/summary.csv"


def canonical_prediction_table(df):
    """Only use explicit label columns; never guess class order from a matrix."""
    actual = next((x for x in ("true_class", "actual", "y_true", "true_label") if x in df), None)
    pred = next((x for x in ("pred_Ensemble_50_50", "predicted_class", "predicted", "y_pred", "predicted_label") if x in df), None)
    if not actual or not pred:
        return None
    data = df.copy()
    data["Gerçek sınıf"] = data[actual].astype(str).str.strip().str.upper()
    data["Tahmin"] = data[pred].astype(str).str.strip().str.upper()
    for field in ("Gerçek sınıf", "Tahmin"):
        data[field] = data[field].replace({"0": "NORMAL", "1": "BACTERIAL", "2": "VIRAL"})
    if not data["Gerçek sınıf"].isin(CLASSES).all() or not data["Tahmin"].isin(CLASSES).all():
        return None
    score_candidates = (
        ("ensemble_prob_normal", "ensemble_prob_bacterial", "ensemble_prob_viral"),
        ("prob_normal", "prob_bacterial", "prob_viral"),
        ("score_NORMAL", "score_BACTERIAL", "score_VIRAL"),
        ("p_normal", "p_bacterial", "p_viral"),
    )
    scores = next((names for names in score_candidates if all(c in data for c in names)), None)
    if scores:
        values = data[list(scores)].apply(pd.to_numeric, errors="coerce")
        if values.notna().all().all() and np.isfinite(values.to_numpy(dtype=float)).all():
            # Skorlar 0–1 olasılık ölçeğinde olmalı; yüzde biçimli girdiyi sessizce kabul etme.
            if ((values >= 0) & (values <= 1)).all().all():
                data["En yüksek skor"] = values.max(axis=1)
    path = next((x for x in ("original_path", "file", "file_path", "image_path", "relative_path") if x in data), None)
    if path:
        data["Dosya"] = data[path].astype(str)
    elif "image_stem" in data:
        data["Görüntü kimliği"] = data["image_stem"].astype(str)
    data["Doğru mu?"] = data["Gerçek sınıf"] == data["Tahmin"]
    return data


def show_confusion_explanation(cm):
    st.caption("**Tabloyu okuma:** Satır gerçek etiketi, sütun model tahminini gösterir. Köşegen üzerindeki değerler doğru, diğerleri hatalı tahminlerdir.")
    pairs = [(int(cm.loc[a, b]), a, b) for a in CLASSES for b in CLASSES if a != b]
    count, actual, predicted = max(pairs)
    if count:
        st.info(f"En sık karışıklık: Gerçekte **{actual}** olan **{count}** görüntü **{predicted}** tahmin edilmiş.")
        st.caption(
            f"**Bu ne demek?** Veri kümesinde {actual} etiketi taşıyan {count} röntgeni "
            f"model {predicted} olarak işaretlemiş. Bu, iki etiketin model tarafından "
            "karıştırıldığını gösterir; görüntünün klinik olarak neden böyle sınıflandırıldığını kanıtlamaz. "
            "Bakteriyel ve viral etken ayrımı tek başına röntgenden kesinleştirilemez."
        )


def show_ensemble_predictions(path, split_name):
    if not path.is_file():
        st.info(f"{split_name} tahmin dosyası bulunamadı. Özet metrikler yine gösterilebilir.")
        return
    raw = load_csv(path)
    data = canonical_prediction_table(raw)
    if data is None:
        st.warning("Tahmin dosyasındaki sınıf sütunları tanınmadı; hatalı analiz üretmemek için ayrıntılar gösterilmiyor.")
        with st.expander("Teknik dosya bilgisi"):
            st.write(list(raw.columns))
        return
    if data.empty:
        st.warning("Tahmin dosyası boş.")
        return
    correct = int(data["Doğru mu?"].sum())
    wrong_all = data.loc[~data["Doğru mu?"]].copy()
    x, y, z = st.columns(3)
    x.metric("Doğru tahmin", correct)
    y.metric("Hatalı tahmin", len(wrong_all))
    z.metric("Doğruluk", tr_percent(correct / len(data)))
    st.caption(f"Bu sayılar {len(data)} kayıt üzerinden hesaplandı; model yeniden çalıştırılmadı.")
    cm = pd.crosstab(data["Gerçek sınıf"], data["Tahmin"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    cm.index.name = "Gerçek sınıf"
    cm.columns.name = "Tahmin edilen sınıf"
    st.markdown("#### Model hangi sınıfları karıştırıyor?")
    st.dataframe(cm, use_container_width=True)
    show_confusion_explanation(cm)
    rows = []
    for label in CLASSES:
        tp = int(cm.loc[label, label])
        support = int(cm.loc[label].sum())
        predicted_count = int(cm[label].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"Sınıf": label, "Görüntü": support, "Kesinlik (%)": 100 * precision, "Duyarlılık (%)": 100 * recall, "F1 (%)": 100 * f1})
    st.markdown("#### Sınıflar ne kadar doğru tanınıyor?")
    st.caption("Duyarlılık: Gerçekte o sınıfa ait görüntülerin doğru tanınma oranı. Kesinlik: Modelin o sınıf dediği görüntülerin ne kadarının gerçekten o sınıf olduğu.")
    st.dataframe(pd.DataFrame(rows).style.format({"Kesinlik (%)": "{:.2f}", "Duyarlılık (%)": "{:.2f}", "F1 (%)": "{:.2f}"}), hide_index=True, use_container_width=True)
    st.markdown("#### Yanlış tahmin örnekleri")
    high_count = int((wrong_all["En yüksek skor"] >= .9).sum()) if "En yüksek skor" in wrong_all else None
    if high_count is not None:
        st.caption(f"Yüksek skorlu hata (≥%90): {high_count} görüntü. Modelin yüksek skor vermesi tahminin doğru olduğunu garanti etmez.")
    with st.expander("Yanlış tahminleri filtrele ve tabloyu aç", expanded=False):
        c1, c2 = st.columns(2)
        actual_filter = c1.selectbox("Gerçek sınıf", ["Tümü"] + CLASSES, key=f"ens_actual_{split_name}")
        pred_filter = c2.selectbox("Tahmin edilen sınıf", ["Tümü"] + CLASSES, key=f"ens_pred_{split_name}")
        wrong = wrong_all.copy()
        if actual_filter != "Tümü":
            wrong = wrong[wrong["Gerçek sınıf"] == actual_filter]
        if pred_filter != "Tümü":
            wrong = wrong[wrong["Tahmin"] == pred_filter]
        if "En yüksek skor" in wrong:
            threshold = st.slider("En düşük tahmin skoru (%)", 0, 100, 0, 5, key=f"ens_conf_{split_name}")
            wrong = wrong[wrong["En yüksek skor"] >= threshold / 100].sort_values("En yüksek skor", ascending=False)
        st.caption(f"Filtreye uyan yanlış tahmin: {len(wrong)}")
        cols = [x for x in ("Gerçek sınıf", "Tahmin", "En yüksek skor", "Dosya", "Görüntü kimliği") if x in wrong]
        st.dataframe(wrong[cols], hide_index=True, use_container_width=True)
        st.download_button("Filtrelenmiş hataları CSV indir", wrong.to_csv(index=False).encode("utf-8-sig"), file_name=f"ensemble_{split_name}_errors.csv", mime="text/csv", key=f"ens_download_{split_name}")
        if wrong.empty:
            st.info("Seçilen filtrelere uygun hata yok.")
        elif "Dosya" in wrong:
            index = st.number_input("Görüntü seç (satır numarası)", min_value=1, max_value=len(wrong), value=1, key=f"ens_row_{split_name}")
            item = wrong.iloc[index - 1]
            left, right = st.columns([1, 1])
            with left:
                image_path = safe_image_path(item["Dosya"])
                if image_path:
                    st.image(str(image_path), width="stretch")
                else:
                    st.info("Görüntü yerel diskte bulunamadı; arşivlenmiş olabilir.")
            with right:
                st.write(f"**Gerçek sınıf:** {item['Gerçek sınıf']}")
                st.write(f"**Model tahmini:** {item['Tahmin']}")
                if "En yüksek skor" in item:
                    st.write(f"**En yüksek skor:** {tr_percent(item['En yüksek skor'])}")
                st.caption(f"Dosya: {item['Dosya']}")
    with st.expander("Veri kaynağı ve teknik ayrıntılar"):
        st.code(str(path.relative_to(ROOT)))
        st.caption("Kayıtlı tahminler kullanılır; eğitim veya yeni tahmin yapılmaz.")

logo_col, title_col = st.columns([1, 12], vertical_alignment="center")
with logo_col:
    if LOGO.exists():
        st.image(str(LOGO), width=74)
with title_col:
    st.title("Pnömoni Sınıflandırması | Hata Analizi")
st.info("Akademik araştırma prototipidir. Model skorları kalibre edilmiş klinik tanı olasılıkları değildir.")

if not REPORT_DIR.is_dir():
    st.warning("Önceki modellere ait rapor klasörü bulunamadı. Nihai ensemble sonuçları varsa yine incelenebilir.")

st.caption("**Bu sayfada neler var?** Önce nihai modelin özetini okuyun; ardından yanlış tahminleri, eski modelleri ve teknik raporları inceleyin.")
tab_home, tab_errors, tab_compare, tab_advanced = st.tabs([
    "1 · Genel Bakış", "2 · Hataları İncele", "3 · Model Karşılaştırması", "4 · İleri Analiz"
])

with tab_home:
    st.subheader("Nihai modelin sonuçları")
    st.write("Nihai yöntem, orijinal X-Ray ve akciğer dışı yoğunluğu %50 azaltılmış görüntüyü iki ayrı MobileNetV2 ile değerlendirir. İki modelin sınıf skorları eşit ağırlıkla birleştirilir.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Doğrulama doğruluğu", "%80,34")
    m2.metric("Makro F1", "%80,49")
    m3.metric("İncelenen görüntü", "875")
    st.info("**Basitçe:** 875 doğrulama görüntüsünün 703'ü doğru, 172'si yanlış sınıflandırıldı. Makro F1, üç sınıfı eşit ağırlıkla değerlendirir.")
    st.markdown("#### Sınıflar ne kadar yakalanıyor?")
    st.caption("Duyarlılık (recall): Gerçekte ilgili sınıfa ait görüntülerin kaçının doğru tahmin edildiği.")
    for label, value in (("NORMAL", .929167), ("BACTERIAL", .745803), ("VIRAL", .775229)):
        st.write(f"**{label}** · {tr_percent(value)}")
        st.progress(value)
    st.warning("Bu sonuçlar kayıtlı doğrulama değerlendirmesidir. Klinik tanı başarısını veya farklı hastanelerdeki performansı kanıtlamaz.")
    with st.expander("Test sonuçları neden ayrıca değerlendirilir?"):
        st.write("Önceden incelenmiş 960 test görüntüsünde 776 doğru ve 184 hatalı tahmin kaydedildi (doğruluk %80,83; makro F1 %80,10). Test kümesi geliştirme sürecinde daha önce incelendiği için tamamen dokunulmamış dış doğrulama olarak sunulamaz.")
        st.write("Conservative Group ise 876 görüntülük ayrı bir retrospektif değerlendirmedir. Farklı veri kümelerindeki sonuçlar doğrudan model üstünlüğü kanıtı olarak sıralanmamalıdır.")
    show_metric_explanations()

with tab_errors:
    st.subheader("Nihai model nerelerde hata yapıyor?")
    st.write("Önce değerlendirme kümesini seçin. Karışıklık matrisi hangi sınıfların birbirine karıştırıldığını, yanlış tahminler ise tekil örnekleri gösterir.")
    split = st.radio("İncelenecek değerlendirme", ["Doğrulama · 875 görüntü", "Önceden incelenmiş test · 960 görüntü"], horizontal=True)
    if split.startswith("Doğrulama"):
        path = next((p for p in ENSEMBLE_VALIDATION_CANDIDATES if p.is_file()), None)
        if path is None:
            st.warning("Nihai ensemble doğrulama tahmin dosyası bulunamadı.")
        else:
            show_ensemble_predictions(path, "validation")
    else:
        st.caption("Bu test kümesi daha önce incelenmiştir; bağımsız dış doğrulama değildir.")
        show_ensemble_predictions(ENSEMBLE_TEST_PREDICTIONS, "test")

with tab_compare:
    st.subheader("Modeller arasında ne fark var?")
    st.caption("Önceki üç model aynı 875 görüntülük doğrulama raporunda değerlendirilmiştir. Nihai ensemble farklı deney kaydından gelir; yöntem ayrıntıları göz önünde tutulmalıdır.")
    if REPORT_DIR.is_dir():
        st.subheader("Önceki üç modelin doğrulama performansı")
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
                # Okunabilir ana tablo: yüzdeler yüzde olarak gösterilir, ham 0–1 değerleri gizlenir.
                main_cols = [c for c in ("model", "n", "accuracy", "macro_f1") if c in summary.columns]
                readable = summary[main_cols].copy()
                readable = readable.rename(columns={"model": "Model", "n": "Görüntü", "accuracy": "Doğruluk (%)", "macro_f1": "Makro F1 (%)"})
                for metric in ("Doğruluk (%)", "Makro F1 (%)"):
                    if metric in readable.columns:
                        readable[metric] = pd.to_numeric(readable[metric], errors="coerce") * 100
                st.markdown("#### Üç modelin sonuçları")
                st.caption("Aşağıdaki yüzdeler aynı 875 görüntülük doğrulama raporundan alınmıştır. En yüksek değerler yalnızca bu üç model arasındadır.")
                st.dataframe(readable.style.format({c: "{:.2f}" for c in ("Doğruluk (%)", "Makro F1 (%)") if c in readable.columns}, na_rep="—"), hide_index=True, use_container_width=True)
                with st.expander("Diğer teknik metrikler (ECE, Brier ve hata sayıları)", expanded=False):
                    display = summary[available].rename(columns=SUMMARY_COLUMNS).copy()
                    formats = {}
                    for original in ("accuracy", "macro_f1", "ece_10_bins", "multiclass_brier"):
                        label = SUMMARY_COLUMNS[original]
                        if label in display.columns:
                            formats[label] = "{:.4f}"
                    st.caption("Teknik tabloda doğruluk ve makro F1 değerleri 0–1 ölçeğindedir (ör. 0,7829 = %78,29).")
                    st.dataframe(display.style.format(formats, na_rep="—"), hide_index=True, use_container_width=True)
            else:
                st.warning("Özet raporun sütunları beklenen biçimde değil.")
        else:
            st.warning("Model karşılaştırma raporu bulunamadı.")
        show_metric_explanations()
        st.caption("ECE ve Brier skorları olasılık kalibrasyonuyla ilişkilidir; yüksek doğruluk tek başına iyi kalibrasyon anlamına gelmez.")
    st.markdown("#### Segmentasyon maskesi sonucu nasıl etkiledi?")
    st.dataframe(pd.DataFrame([
        {"Yöntem": "A · Orijinal görüntü", "Doğruluk (%)": 79.89, "Makro F1 (%)": 79.75},
        {"Yöntem": "B · Sert maske", "Doğruluk (%)": 66.74, "Makro F1 (%)": 66.75},
        {"Yöntem": "Nihai · Orijinal + B3 ensemble", "Doğruluk (%)": 80.34, "Makro F1 (%)": 80.49},
    ]).style.format({"Doğruluk (%)": "{:.2f}", "Makro F1 (%)": "{:.2f}"}), hide_index=True, use_container_width=True)
    st.info("Sert maske, akciğer dışını tamamen sıfırlar ve bu deneyde doğruluğu **13,15 yüzde puan düşürür**. Nihai yöntemde akciğer dışı yoğunluk yarıya indirilir ve iki model birleştirilir. Bu nedenle nihai kazanım yalnızca maske etkisi olarak yorumlanamaz.")
    st.caption("**Karşılaştırma sınırı:** Önceki üç modelin raporu ile A/B/ensemble deneyleri farklı değerlendirme kayıtlarından gelir. Nihai ensemble'ın %80,34 sonucunu, önceki Fine-Tuned modelin %78,29 sonucuyla kesin üstünlük kanıtı olarak yorumlamayın; eşleştirilmiş örnek ve protokol kontrolü gerekir.")

with tab_advanced:
    st.subheader("Ayrıntılı araştırma raporları")
    st.caption("Bu bölüm, önceki modellerin kalibrasyonunu ve Conservative Group'un ayrı testini incelemek isteyenler içindir.")
    with st.expander("Önceki modeller · Hatalı tahminler", expanded=False):
        if REPORT_DIR.is_dir():
            model_name = st.selectbox("İncelenecek önceki model", list(MODEL_FILES), key="legacy_error_model")
            prefix = MODEL_FILES[model_name]
            error_path = REPORT_DIR / f"{prefix}_errors.csv"
            if not error_path.is_file():
                st.info(
                    "Bu modelin örnek düzeyindeki hata kayıtları bu kopyada bulunmuyor. "
                    "Hata tablosu, örnek görüntü incelemesi ve filtrelenmiş hata indirmesi "
                    "kullanılamıyor; toplu analizleri diğer bölümlerde inceleyebilirsiniz."
                )
            else:
                errors = load_csv(error_path)
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
        else:
            st.info("Örnek düzeyindeki hata kayıtları bu kopyada bulunmuyor.")
    with st.expander("Önceki modeller · Karışıklık matrisi ve kalibrasyon", expanded=False):
        if REPORT_DIR.is_dir():
            model_name = st.selectbox("İncelenecek önceki model", list(MODEL_FILES), key="legacy_cal_model")
            prefix = MODEL_FILES[model_name]
            cm_path = REPORT_DIR / f"{prefix}_confusion_matrix.csv"
            cal_path = REPORT_DIR / f"{prefix}_calibration.csv"
            missing = [p.name for p in (cm_path, cal_path) if not p.is_file()]
            if missing:
                st.warning(
                    "Bu modelin toplu karışıklık matrisi veya kalibrasyon kaydı eksik. "
                    "Bu bölüm kullanılamıyor; diğer analizlere devam edebilirsiniz."
                )
            else:
                cal = load_csv(cal_path)
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
                    st.dataframe(cm.rename(columns={"Unnamed: 0": "Gerçek sınıf"}), use_container_width=True)
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
                        st.caption(f"**Nasıl yorumlanır?** {real_cls} etiketli {count} görüntüye model {predicted_cls} etiketi vermiş. Bu, veri kümesi etiketlerine göre bir sınıflandırma hatasıdır; klinik etkeni veya hata nedenini tek başına açıklamaz.")

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
        else:
            st.info("Toplu karışıklık matrisi ve kalibrasyon kayıtları bu kopyada bulunmuyor.")
    with st.expander("Conservative Group · 876 görüntülük ayrı test", expanded=False):
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
st.caption("Bu sayfa yalnızca kayıtlı raporları okur. Model eğitimi veya yeni tahmin gerçekleştirmez. Model skorları klinik tanı olasılığı değildir.")
