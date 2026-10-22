# -*- coding: utf-8 -*-
"""Shared presentation helpers. Does not load or change models."""
from pathlib import Path
from html import escape
import streamlit as st

def apply_extended_ui():
    st.markdown("""<style>
    .pa-hero {display:flex;align-items:center;gap:24px;background:linear-gradient(110deg,#fff 60%,#DDF4F4);
      border:1px solid #D7E9E9;border-radius:22px;padding:27px 30px;margin:6px 0 17px;
      box-shadow:0 10px 32px rgba(20,83,89,.045)}
    .pa-hero img {width:76px;height:76px;object-fit:contain;border-radius:16px}
    .pa-hero .pa-kicker {font-size:10px;letter-spacing:.2em;color:#078E91;font-weight:800}
    .pa-hero h1 {font-size:35px!important;margin:9px 0 6px!important;color:#123944!important;line-height:1.12}
    .pa-hero p {margin:0;color:#64838A;font-size:14px}
    .pa-hero .pa-badges {margin-top:16px;display:flex;gap:8px;flex-wrap:wrap}
    .pa-hero .pa-badges span {background:#EAF8F5;border:1px solid #D0EAE7;border-radius:30px;
      padding:6px 10px;font-size:11px;font-weight:700;color:#277A78}
    .pa-intro {background:#fff;border:1px solid #DCEBEA;border-radius:16px;padding:17px 20px;
      margin:12px 0 18px;color:#466D76;font-size:14px;line-height:1.8}
    .pa-intro strong {color:#123944}
    [data-testid="stFileUploader"] {background:#fff;border:1px solid #DCEBEA;border-radius:15px;padding:12px}
    [data-testid="stFileUploaderDropzone"] {background:#F8FCFB;border:1px dashed #9FCFCC;border-radius:10px}
    [data-testid="stExpander"] {border:1px solid #D9E9E7;border-radius:12px}
    @media(max-width:650px) {.pa-hero{padding:20px;gap:13px}.pa-hero img{width:54px;height:54px}
      .pa-hero h1{font-size:26px!important}}
    </style>""",unsafe_allow_html=True)

def hero(title, subtitle, badges=(), root=None):
    # Local image embedded as a data URI to keep the hero stable on every Streamlit page.
    import base64
    root = Path(root) if root else Path.cwd()
    logo = root / "assets" / "pneumonia_ai_logo.png"
    if logo.is_file():
        data = base64.b64encode(logo.read_bytes()).decode("ascii")
        img = f'<img src="data:image/png;base64,{data}" alt="Pneumonia AI logo">'
    else:
        img = '<div style="font-size:48px">🫁</div>'
    tags = "".join(f"<span>{escape(str(x))}</span>" for x in badges)
    st.markdown(f"""<div class="pa-hero">{img}<div>
      <div class="pa-kicker">TIBBİ GÖRÜNTÜLEME / ARAŞTIRMA PLATFORMU</div>
      <h1>{escape(title)}</h1><p>{escape(subtitle)}</p>
      <div class="pa-badges">{tags}</div></div></div>""",unsafe_allow_html=True)

MODEL_DESCRIPTIONS = {
"Baseline CNN": """**Çalışma ilkesi:** Evrişim katmanları görüntüde kenar, doku ve daha karmaşık örüntüleri öğrenir. Havuzlama veya benzeri boyut azaltma işlemleri özellik haritalarını sıkıştırır; son sınıflandırma bölümü NORMAL, BACTERIAL ve VIRAL sınıfları için skor üretir.

**Araştırmadaki rolü:** Daha karmaşık veya önceden eğitilmiş mimarilere karşı başlangıç (baseline) karşılaştırması sağlar. Başarısı eğitim verisinin miktarına, sınıf dağılımına ve hiperparametrelere bağlıdır.

**Sınırlama:** Görüntüdeki cihaz, işaretleme ve veri kaynağı farklılıklarını hastalıkla ilişkili özelliklerle karıştırabilir. Mimari katman sayısı ve filtre adetleri kayıtlı modelden doğrulanmalıdır.""",
"MobileNetV2": """**Çalışma ilkesi:** MobileNetV2, hesaplama maliyetini azaltmayı hedefleyen *depthwise separable convolution* yaklaşımından yararlanır. *Inverted residual* ve *linear bottleneck* blokları, özellikleri görece verimli biçimde işlemeye yardımcı olur.

**Transfer öğrenme:** Önceden öğrenilmiş görsel temsiller, akciğer X-Ray sınıflandırmasına uyarlanabilir. Bu yaklaşım küçük veya orta ölçekli veri kümelerinde sıfırdan eğitime alternatif oluşturur.

**Sınırlama:** Doğal görüntülerden öğrenilen özellikler tıbbi görüntülere kusursuz aktarılmaz; veri dağılımı değişimlerinde performans düşebilir. Projede hangi katmanların eğitildiği eğitim kayıtlarıyla doğrulanmalıdır.""",
"Fine-Tuned MobileNetV2": """**Çalışma ilkesi:** Transfer öğrenme sonrasında ağın belirli katmanlarının yeniden eğitilmesi (*fine-tuning*), özelliklerin hedef veri kümesine daha iyi uyarlanmasını amaçlar. Açılan katmanlar ve öğrenme oranı modelin ne kadar değişeceğini etkiler.

**Araştırmadaki rolü:** Sadece özellik çıkarma yaklaşımına kıyasla hedef veri kümesine uyarlamanın katkısını incelemeyi sağlar.

**Sınırlama:** Fazla katman açmak veya yüksek öğrenme oranı kullanmak aşırı öğrenmeye ve önceden öğrenilen temsillerin bozulmasına neden olabilir. Kesin fine-tuning stratejisi deney kayıtlarından teyit edilmelidir.""",
"MobileNetV2 - Conservative Group": """**Çalışma ilkesi:** MobileNetV2 tabanlı ayrı bir deney koludur. 'Conservative Group' adı tek başına özgül eğitim protokolünü veya karar eşiğini açıklamaz; bunlar deney yapılandırmasından doğrulanmalıdır.

**Değerlendirme:** Bu deneyin raporlanan sonuçları 876 görüntülük ayrı retrospektif test kümesinden gelir. Diğer üç modelin 875 görüntülük doğrulama sonuçlarıyla aynı koşullarda elde edilmiş gibi doğrudan sıralanmamalıdır.

**Sınırlama:** Test kümesinin örnekleme yapısı ve dağılımı sonuçları etkiler; tek bir test başarısı klinik genellenebilirlik kanıtı değildir.""",
"U-Net V3": """**Çalışma ilkesi:** U-Net ailesi, *encoder–decoder* düzeniyle görüntüden çok ölçekli özellikler çıkarır ve bunları piksel düzeyindeki bir maskeye dönüştürür. *Skip connection* bağlantıları, daha erken katmanlardaki mekânsal ayrıntıların geri kazanılmasına yardımcı olur.

**Projede kullanımı:** Görüntü 256×256 gri tona dönüştürülüp [0,1] aralığına ölçeklenir. Model çıktısına 0,5 eşik uygulanarak ikili maske oluşturulur; maske orijinal görüntü boyutuna taşınır.

**Sınırlama:** Maske model tahminidir, uzman tarafından çizilmiş referans anotasyon değildir. Yoğun opasite, örtüşme ve görüntü kalitesi sınırların hatalı belirlenmesine yol açabilir.""",
"Deney A - Orijinal": """**Amaç:** Sınıflandırıcıyı orijinal X-Ray görüntüsüyle değerlendirerek maskeleme uygulanmayan referans deney kolunu oluşturur.

**İşlem:** RGB görüntü 224×224 boyutuna dönüştürülür ve ilgili sınıflandırma modeline verilir.

**Yorum:** Sonuç, yalnızca modelin ürettiği üç sınıf skorunu gösterir; tek başına klinik tanı değildir.""",
"Deney B - Maskeli": """**Amaç:** Akciğer dışındaki bölgelerin maskelenmesinin sınıflandırma çıktısına etkisini incelemektir.

**İşlem:** U-Net V3'ün ürettiği ikili maske orijinal görüntüye uygulanır; maske dışı pikseller sıfırlanır ve görüntü 224×224 boyutuna getirilir.

**Sınırlama:** Segmentasyon hataları sınıflandırmaya aktarılabilir. Dolayısıyla maskeleme otomatik olarak daha iyi başarı anlamına gelmez."""
}

def model_guide(names=None, title="Modellerin çalışma prensipleri ve sınırlamaları"):
    names = names or list(MODEL_DESCRIPTIONS)
    st.markdown(f"### {title}")
    st.caption("Genel mimari açıklamaları ile bu uygulamanın kodundan doğrulanabilen işlemler ayrılmıştır.")
    for name in names:
        if name in MODEL_DESCRIPTIONS:
            with st.expander(name):
                st.markdown(MODEL_DESCRIPTIONS[name])

def intro(html):
    st.markdown(f'<div class="pa-intro">{html}</div>', unsafe_allow_html=True)
