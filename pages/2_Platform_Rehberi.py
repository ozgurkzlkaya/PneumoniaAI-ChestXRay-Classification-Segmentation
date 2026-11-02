from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st
from medical_theme import apply_medical_theme, render_medical_sidebar

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "assets" / "pneumonia_ai_logo.png"
st.set_page_config(
    page_title="Pneumonia AI | Platform Rehberi",
    page_icon=str(LOGO) if LOGO.exists() else "🫁",
    layout="wide",
)
render_medical_sidebar()
apply_medical_theme()

CLASSES = ["NORMAL", "BACTERIAL", "VIRAL"]
SPLITS = pd.DataFrame({
    "Sınıf": CLASSES,
    "Eğitim": [1089, 1878, 1054],
    "Doğrulama": [240, 417, 218],
    "Test": [254, 485, 221],
})
METRICS = [
    ("Doğruluk (Accuracy)", "Tüm görüntüler arasında doğru tahmin edilenlerin oranıdır. Sınıflar dengesizse tek başına yeterli değildir."),
    ("Kesinlik (Precision)", "Modelin belirli bir sınıf dediği görüntülerin ne kadarının gerçekten o sınıfta etiketlendiğidir."),
    ("Duyarlılık (Recall)", "Gerçekte bir sınıfa ait görüntülerin ne kadarının doğru tanındığıdır."),
    ("F1 skoru", "Kesinlik ve duyarlılığı birlikte özetleyen harmonik ortalamadır."),
    ("Makro F1", "Üç sınıfın F1 skorlarının eşit ağırlıklı ortalamasıdır."),
    ("Dengeli doğruluk", "Sınıfların duyarlılıklarının aritmetik ortalamasıdır."),
    ("Karışıklık matrisi", "Satırlar gerçek etiketi, sütunlar model tahminini gösterir. Köşegen doğru, diğer hücreler yanlış tahminlerdir."),
    ("ROC-AUC", "Bir sınıfı diğerlerinden ayırma becerisini farklı eşikler üzerinden özetler. %90 AUC, %90 doğruluk değildir."),
    ("Kesinlik–duyarlılık eğrisi", "Karar eşiği değiştiğinde kesinlik ile duyarlılığın nasıl değiştiğini gösterir."),
    ("ECE", "Tahmin skoruyla gözlenen doğruluk arasındaki uyumsuzluğun özetidir; hesaplama aralıklarına duyarlıdır."),
    ("Brier skoru", "Tahmin edilen sınıf skorları ile gerçek etiketler arasındaki karesel hata ölçüsüdür. Düşük değer tercih edilir."),
]

head_left, head_right = st.columns([1, 11], vertical_alignment="center")
with head_left:
    if LOGO.exists():
        st.image(str(LOGO), width=72)
with head_right:
    st.title("Pneumonia AI | Platform Rehberi")
    st.caption("Projeyi tanıyın, yöntemi öğrenin ve araştırma sonuçlarını doğru yorumlayın.")

st.info("Bu platform akademik araştırma prototipidir; tıbbi tanı, triyaj veya tedavi amacıyla kullanılmamalıdır.")
st.markdown("**Bu rehber nasıl okunur?** İlk bölümden başlayarak projenin amacını, verileri, çalışma yöntemini ve sonuçlarını adım adım inceleyebilirsiniz. Teknik ayrıntılar açılır alanlarda tutulmuştur.")

tabs = st.tabs([
    "1 · Projeyi Tanıyın",
    "2 · Veri Kümesi",
    "3 · Nasıl Çalışıyor?",
    "4 · Araştırma Sonuçları",
    "5 · Kullanım Rehberi",
    "6 · Bilimsel Sınırlılıklar",
])

with tabs[0]:
    st.subheader("Bu proje ne yapıyor?")
    st.write("Pneumonia AI, akciğer röntgenlerini veri kümesindeki **NORMAL**, **BACTERIAL** ve **VIRAL** etiketlerine göre sınıflandıran modelleri araştırmak için geliştirilmiştir. Ayrıca akciğer bölgesini tahmin eden bir segmentasyon modeli ve kayıtlı deneylerin hata analizleri bulunur.")
    a, b, c, d = st.columns(4)
    a.metric("Sınıf", "3")
    b.metric("Karşılaştırma modeli", "4")
    c.metric("Ablasyon kolu", "2")
    d.metric("Segmentasyon", "U-Net V3")
    st.markdown("#### Projenin gelişim süreci")
    st.markdown(
        "**1. Baseline CNN:** Temel karşılaştırma modeli oluşturuldu.\n\n"
        "**2. MobileNetV2:** Transfer öğrenmeyle önceden öğrenilmiş görüntü özelliklerinden yararlanıldı.\n\n"
        "**3. Fine-Tuned MobileNetV2:** Bazı önceden eğitilmiş katmanlar da veriye uyarlanarak yeniden eğitildi.\n\n"
        "**4. U-Net V3:** Pnömoni sınıfı yerine akciğer bölgelerini tahmin eden segmentasyon modeli geliştirildi. Nihai yöntemin B3 kolunda bu maske, akciğer dışındaki alanların etkisini azaltmak için kullanıldı.\n\n"
        "**5. Original + B3 ensemble:** Orijinal ve yumuşak maskelenmiş görüntülerden gelen iki model skoru birleştirildi."
    )
    st.caption("Bu aşamalar farklı deney kayıtlarını temsil eder. Her yeni aşamanın tek başına klinik iyileşme sağladığı sonucu çıkarılamaz.")
    with st.expander("Etiketlerin klinik anlamı hakkında önemli not"):
        st.write("BACTERIAL ve VIRAL burada veri kümesinin araştırma etiketleridir. Tek bir akciğer röntgeninden enfeksiyon etkeni kesin olarak belirlenemez; uzman değerlendirmesi ve gerektiğinde ek testler gerekir.")

with tabs[1]:
    st.subheader("Hangi görüntüler kullanıldı?")
    a, b, c, d = st.columns(4)
    a.metric("Eğitim", "4.021")
    b.metric("Doğrulama", "875")
    c.metric("Test", "960")
    d.metric("Toplam", "5.856")
    st.write("Eğitim, doğrulama ve test görüntüleri model geliştirme ve değerlendirme süreçlerinde farklı amaçlarla kullanılır. Bu sayılar kayıtlı veri bölünmelerini gösterir.")
    st.bar_chart(SPLITS.set_index("Sınıf")[["Eğitim", "Doğrulama", "Test"]], use_container_width=True)
    st.dataframe(SPLITS, hide_index=True, use_container_width=True)
    with st.expander("NORMAL, BACTERIAL ve VIRAL ne demek?", expanded=True):
        st.markdown(
            "**NORMAL:** Veri kümesinde pnömoni etiketi bulunmayan görüntüler.\n\n"
            "**BACTERIAL:** Bakteriyel pnömoni olarak etiketlenmiş görüntüler.\n\n"
            "**VIRAL:** Viral pnömoni olarak etiketlenmiş görüntüler."
        )
        st.caption("Etiketler klinik tanı garantisi değildir.")
    st.warning("Conservative Group için raporlanan 876 görüntülük retrospektif test, yukarıdaki 960 görüntülük test raporundan farklı bir değerlendirmedir. 5.856 toplamına ikinci kez eklenmez.")

with tabs[2]:
    st.subheader("Bir görüntü nasıl analiz ediliyor?")
    st.markdown(
        "**1. Görüntü yüklenir.** Kullanıcı akciğer X-Ray dosyasını seçer.\n\n"
        "**2. Ön işleme yapılır.** Görüntü, seçilen modelin beklediği biçime getirilir.\n\n"
        "**3. Akciğer bölgesi tahmin edilir.** U-Net V3, segmentasyon maskesi üretir.\n\n"
        "**4. Sınıflandırma yapılır.** Modeller üç araştırma etiketi için skor üretir.\n\n"
        "**5. Sonuçlar incelenir.** Sınıf skorları, segmentasyon ve kayıtlı hata analizleri değerlendirilir."
    )
    st.markdown("#### Nihai model nasıl çalışır?")
    st.write("Nihai sistem **iki ayrı fine-tuned MobileNetV2** kullanır. Birinci model orijinal röntgeni, ikinci model ise akciğer dışındaki görüntü yoğunluğu azaltılmış sürümü değerlendirir. İki modelin üç sınıfa verdiği skorlar eşit ağırlıkla ortalanır.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original kolu**")
        st.write("Orijinal RGB görüntü → 224×224 yeniden boyutlandırma → fine-tuned MobileNetV2.")
    with right:
        st.markdown("**B3 kolu**")
        st.write("U-Net V3 maskesi → maske düzeltme → akciğer dışı yoğunluğu %50'ye düşürme → B3 fine-tuned MobileNetV2.")
    st.code("p_final = 0.50 * p_original + 0.50 * p_B3", language="python")
    st.caption("Bu bir çoğunluk oylaması değil, üç sınıfa ait softmax skorlarının ağırlıklı ortalamasıdır. Skorlar kalibre edilmiş klinik olasılıklar değildir.")
    with st.expander("Teknik ön işleme ayrıntıları"):
        st.markdown(
            "**Original:** RGB X-Ray, 224×224 bilinear yeniden boyutlandırma.\n\n"
            "**B3:** Gri 256×256 görüntü, U-Net V3 maskesi, 0,5 eşik ve rafinasyon; maske orijinal boyuta en yakın komşu yöntemiyle aktarılır. Akciğer dışındaki RGB piksel yoğunluğu 0,50 ile çarpılır. Son görüntü 224×224 boyuta getirilir.\n\n"
            "**Segmentasyon sınırı:** Maske tahminidir, uzman onaylı referans anotasyon değildir. Özellikle yoğun opasitelerde eksik veya hatalı bölgeler olabilir."
        )
    with st.expander("Diğer modeller ne işe yarıyor?"):
        st.markdown(
            "**Baseline CNN:** Başlangıç karşılaştırması.\n\n"
            "**MobileNetV2:** Transfer öğrenme modeli.\n\n"
            "**Fine-Tuned MobileNetV2:** Önceden eğitilmiş katmanların bir kısmı da uyarlanmış sürüm.\n\n"
            "**Conservative Group:** Ayrı deney kapsamında değerlendirilmiş MobileNetV2 modelidir. 876 görüntülük retrospektif test sonuçları, diğer modellerin doğrulama sonuçlarıyla doğrudan sıralanamaz.\n\n"
            "**U-Net V3:** Akciğer maskesi tahmini; sınıflandırma modelinden farklı görev."
        )

with tabs[3]:
    st.subheader("Araştırma ne sonuç verdi?")
    st.write("Aşağıdaki sayılar kayıtlı değerlendirme raporlarından alınmıştır. Bu rehber yeni eğitim veya tahmin çalıştırmaz.")
    a, b, c = st.columns(3)
    a.metric("Nihai doğrulama doğruluğu", "%80,34")
    b.metric("Nihai makro F1", "%80,49")
    c.metric("Doğru / toplam", "703 / 875")
    st.markdown("**Basit yorum:** Doğrulama kümesindeki 875 görüntünün 703'ü veri kümesinin etiketiyle aynı sınıfta, 172'si farklı sınıfta tahmin edildi.")
    st.markdown("#### Nihai modelin sınıflara göre duyarlılığı")
    recall = pd.DataFrame({
        "Sınıf": CLASSES,
        "Doğrulama (%)": [92.92, 74.58, 77.52],
        "Önceden incelenmiş test (%)": [95.28, 77.94, 70.59],
    })
    recall_long = recall.melt(
        id_vars="Sınıf",
        value_vars=["Doğrulama (%)", "Önceden incelenmiş test (%)"],
        var_name="Değerlendirme",
        value_name="Duyarlılık (%)",
    )
    recall_chart = (
        alt.Chart(recall_long)
        .mark_bar()
        .encode(
            x=alt.X("Sınıf:N", sort=CLASSES, title="Sınıf"),
            xOffset=alt.XOffset("Değerlendirme:N"),
            y=alt.Y("Duyarlılık (%):Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]), title="Duyarlılık (%)"),
            color=alt.Color("Değerlendirme:N", title="Değerlendirme"),
            tooltip=[
                alt.Tooltip("Sınıf:N"),
                alt.Tooltip("Değerlendirme:N"),
                alt.Tooltip("Duyarlılık (%):Q", format=".2f"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(recall_chart, use_container_width=True)
    st.caption("Duyarlılık: Gerçekte ilgili etiketi taşıyan görüntülerin doğru tanınma oranı. VIRAL sınıfındaki test duyarlılığı daha düşüktür.")
    with st.expander("Ablasyon: Orijinal görüntü ve maskeleme karşılaştırması", expanded=True):
        st.dataframe(pd.DataFrame([
            {"Deney": "A · Orijinal görüntü", "Küme": "Doğrulama", "Görüntü": 875, "Doğruluk (%)": 79.89, "Makro F1 (%)": 79.75},
            {"Deney": "B · Sert maske", "Küme": "Doğrulama", "Görüntü": 875, "Doğruluk (%)": 66.74, "Makro F1 (%)": 66.75},
            {"Deney": "Original + B3 ensemble", "Küme": "Doğrulama", "Görüntü": 875, "Doğruluk (%)": 80.34, "Makro F1 (%)": 80.49},
        ]), hide_index=True, use_container_width=True)
        st.write("Sert maske, orijinal deneye göre **13,15 yüzde puan** daha düşük doğruluk verdi. Nihai ensemble orijinal deneyden yaklaşık **0,45 yüzde puan** yüksek çıktı. Ancak B3 yumuşak maske ile iki modelin skor ortalaması birlikte değiştiği için kazanım yalnızca maske etkisine bağlanamaz.")
    with st.expander("Diğer kayıtlı model sonuçları"):
        st.dataframe(pd.DataFrame([
            {"Model": "Baseline CNN", "Değerlendirme": "875 doğrulama", "Doğruluk (%)": 77.71, "Makro F1 (%)": 74.93},
            {"Model": "MobileNetV2", "Değerlendirme": "875 doğrulama", "Doğruluk (%)": 77.03, "Makro F1 (%)": 77.05},
            {"Model": "Fine-Tuned MobileNetV2", "Değerlendirme": "875 doğrulama", "Doğruluk (%)": 78.29, "Makro F1 (%)": 78.45},
            {"Model": "Conservative Group", "Değerlendirme": "876 ayrı test", "Doğruluk (%)": 78.77, "Makro F1 (%)": 78.59},
            {"Model": "Nihai ensemble", "Değerlendirme": "960 incelenmiş test", "Doğruluk (%)": 80.83, "Makro F1 (%)": 80.10},
        ]), hide_index=True, use_container_width=True)
        st.caption("Farklı veri kümeleri ve deney protokolleri doğrudan model üstünlüğü karşılaştırması için eşdeğer değildir.")
    with st.expander("Metrik sözlüğü: teknik kavramlar"):
        for name, explanation in METRICS:
            st.markdown(f"**{name}:** {explanation}")
    st.caption("Detaylı hata tabloları, karışıklık matrisleri ve kalibrasyon grafikleri için Hata Analizi sayfasını kullanın.")

with tabs[4]:
    st.subheader("Uygulamayı nasıl kullanabilirsiniz?")
    st.markdown(
        "**Görüntü Analizi:** X-Ray yükleyin, nihai tahmin ve segmentasyon sonucunu görüntüleyin. Diğer modelleri ve ablasyon çıktılarını karşılaştırın.\n\n"
        "**Hata Analizi:** Önce Genel Bakış ile başlayın. Hataları İncele bölümünde doğrulama/test seçin, karışıklık matrisini okuyun ve yanlış tahminleri filtreleyin. Model Karşılaştırması ve İleri Analiz bölümlerinde önceki deneyleri inceleyin.\n\n"
        "**Platform Rehberi:** Yöntemin nasıl çalıştığını, kullanılan veri kümesini ve değerlendirme sınırlılıklarını öğrenin."
    )
    st.markdown("#### Bir tahmin ekranında nelere bakmalıyım?")
    st.markdown(
        "**Tahmin edilen etiket:** Modelin en yüksek skoru verdiği sınıftır.\n\n"
        "**Üç sınıf skoru:** Model çıktılarının dağılımını gösterir; tanı olasılığı olarak yorumlanmamalıdır.\n\n"
        "**Segmentasyon maskesi:** Modelin tahmin ettiği akciğer bölgesini gösterir; tıbbi anotasyon değildir.\n\n"
        "**Hata Analizi:** Tekil tahminin güvenilirliğini garanti etmez, geçmiş kayıtlı örneklerdeki hata örüntülerini gösterir."
    )
    st.info("Araştırma amaçlı kullanımda görüntünün kaynağını, etiket bilgisini, model sürümünü ve değerlendirme protokolünü birlikte kaydetmek yeniden üretilebilirlik için önemlidir.")

with tabs[5]:
    st.subheader("Sonuçlar ne kadar güvenilir yorumlanabilir?")
    st.error("Bu sistem klinik kullanıma onaylı değildir. Tek başına pnömoni tanısı koymaz, bakteriyel/viral etkeni kesinleştirmez ve hasta yönetiminde karar desteği olarak kullanılmamalıdır.")
    st.markdown("**Veri kümesi sınırlılıkları:** Görüntülerin kaynağı, cihazı, hasta profili ve kalitesi değiştiğinde performans farklılaşabilir. Retrospektif doğruluk, dış merkezde klinik başarıyı kanıtlamaz.")
    st.markdown("**Değerlendirme sınırı:** Doğrulama kümesi geliştirme boyunca tekrar kullanılmıştır; 960 görüntülük test kümesi de daha önce incelenmiştir. Bu nedenle sonuçlar tamamen dokunulmamış, bağımsız dış doğrulama olarak sunulmaz.")
    with st.expander("Veri bütünlüğü denetiminin bulguları", expanded=True):
        st.markdown(
            "- Bölünmeler arasında **1 tam eşleşen görüntü grubu**.\n"
            "- Aynı bölünme içinde **29 yinelenen görüntü grubu**.\n"
            "- Bölünmeler arasında **52 olası hasta kimliği örtüşmesi adayı**.\n"
            "- **51 görüntü yakalama adayı**.\n\n"
            "Son iki bulgu doğrulanmış hasta çakışması değildir; ayrıca incelenmesi gereken adaylardır. "
            "Aynı veya ilişkili görüntülerin farklı veri bölünmelerinde bulunması, ölçülen model başarısını olduğundan yüksek gösterebilir. "
            "Bu nedenle veri bütünlüğü bulguları değerlendirme sonuçlarının yorumlanmasında dikkate alınmalıdır."
        )
    with st.expander("Yeniden üretilebilirlik ve rapor kaynakları"):
        st.markdown(
            "**Kayıtlı nihai doğrulama tahminleri:** `results/classification/ablation_uniform_evaluation/ensemble_50_50/ensemble_predictions.csv`\n\n"
            "**Kayıtlı nihai test tahminleri:** `results/classification/fixed_ensemble_test_evaluation/test_predictions.csv`\n\n"
            "Model ağırlıkları, veri bölünme manifestleri, ön işleme kodları ve deney ayarları sonuçlarla birlikte korunmalıdır. Arşive taşınan veri kopyaları nedeniyle eski betikler doğrudan çalışmayabilir; bu tek başına kayıtlı raporları geçersiz kılmaz."
        )
    st.warning("875 doğrulama, 960 test ve Conservative Group'un 876 örneklik ayrı testi aynı değerlendirme değildir. Farklı raporlar yalnızca aynı örnekler ve aynı protokol doğrulandığında doğrudan karşılaştırılabilir.")
    st.caption("Bunlar araştırma sonuçlarıdır; uzman hekim değerlendirmesinin veya prospektif, hasta bazında ayrılmış dış doğrulamanın yerini tutmaz.")

st.caption("Pneumonia AI · Akademik araştırma platformu · Bu rehber kayıtlı bulguları açıklar; yeni model eğitimi veya çıkarım çalıştırmaz.")
