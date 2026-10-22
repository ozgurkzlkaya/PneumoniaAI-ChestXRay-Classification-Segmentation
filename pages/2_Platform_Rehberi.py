"""Pneumonia AI | Proje Hakkında. Yalnızca akademik araştırma amaçlıdır."""
from pathlib import Path
import base64
import html
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / 'assets' / 'pneumonia_ai_logo.png'
st.set_page_config(page_title='Pneumonia AI | Proje Hakkında', page_icon=str(LOGO) if LOGO.exists() else '🫁', layout='wide', initial_sidebar_state='expanded')

try:
    from medical_theme import apply_medical_theme, render_medical_sidebar
except ImportError:
    apply_medical_theme = None
    render_medical_sidebar = None

if apply_medical_theme:
    apply_medical_theme()
if render_medical_sidebar:
    render_medical_sidebar()
else:
    with st.sidebar:
        if LOGO.exists():
            st.image(str(LOGO), width=92)
        st.markdown('### Pneumonia AI')
        st.caption('TIBBİ GÖRÜNTÜLEME ARAŞTIRMASI')
        st.page_link('app.py', label='Görüntü Analizi', icon=':material/home:')
        st.page_link('pages/1_Hata_Analizi.py', label='Hata Analizi', icon=':material/analytics:')
        st.page_link('pages/2_Platform_Rehberi.py', label='Proje Hakkında', icon=':material/menu_book:')
        st.caption('Akademik araştırma prototipi. Klinik tanı aracı değildir.')

logo_b64 = base64.b64encode(LOGO.read_bytes()).decode('ascii') if LOGO.exists() else ''
logo_html = f'<img class="pa-logo" src="data:image/png;base64,{logo_b64}" alt="Pneumonia AI logosu">' if logo_b64 else '<div class="pa-logo-placeholder">🫁</div>'

st.markdown('''<style>
:root { --pa-ink:#0b3446; --pa-teal:#008f92; --pa-muted:#6f8795; --pa-line:#dcebee; }
[data-testid="stAppViewContainer"] { background:linear-gradient(118deg,#f6fbfc 0%,#f1f9fa 72%,#e9f7f6 100%); }
[data-testid="stMainBlockContainer"] {max-width:1440px;padding-top:2.3rem;padding-bottom:5rem;}
.pa-hero {background:linear-gradient(110deg,#ffffff 0%,#f4fbfd 54%,#d6eff2 100%);border:1px solid #dcebef;border-radius:22px;padding:26px 32px;box-shadow:0 10px 34px rgba(23,84,98,.055);display:flex;gap:23px;align-items:center;margin-bottom:16px;}
.pa-logo {width:90px;height:90px;object-fit:contain;border-radius:20px;filter:drop-shadow(0 5px 9px rgba(0,83,91,.12));}
.pa-logo-placeholder {font-size:50px;}
.pa-eyebrow {font-size:11px;letter-spacing:2.6px;color:#008f92;font-weight:800;margin-bottom:8px;}
.pa-hero h1 {font-size:clamp(30px,3.1vw,47px);line-height:1.1;letter-spacing:-1.2px;color:#0b3446;margin:0 0 8px;font-weight:800;}
.pa-sub {color:#678493;font-size:15px;margin:0;}
.pa-hero-right {margin-left:auto;max-width:320px;color:#195c6c;border-left:3px solid #009da1;padding-left:17px;font-size:15px;line-height:1.55;}
.pa-panel {background:rgba(255,255,255,.87);border:1px solid #dcebef;border-radius:16px;padding:20px 22px;height:100%;box-shadow:0 3px 13px rgba(30,84,98,.035);}
.pa-panel h3 {font-size:18px;margin:0 0 11px;color:#0b3446;}
.pa-panel p,.pa-panel li {color:#5a7482;font-size:14px;line-height:1.65;}
.pa-panel p {margin:0;}
.pa-section {font-size:24px;font-weight:800;letter-spacing:-.4px;color:#0b3446;margin:24px 0 8px;}
.pa-desc {font-size:14px;color:#79909c;margin:0 0 18px;line-height:1.6;}
.pa-kpi {border:1px solid #dcebef;background:#fff;border-radius:14px;padding:15px 17px;min-height:110px;}
.pa-kpi-label {font-size:12px;color:#67828e;}
.pa-kpi-value {font-size:27px;font-weight:800;color:#007f83;margin-top:6px;}
.pa-kpi-note {font-size:12px;color:#80939d;}
.pa-flow {text-align:center;background:#fff;border:1px solid #dcebef;border-radius:15px;padding:18px 10px;min-height:164px;}
.pa-flow-number {display:inline-flex;align-items:center;justify-content:center;background:#078e9e;color:white;width:30px;height:30px;border-radius:50%;font-weight:800;font-size:13px;}
.pa-flow-title {font-size:15px;font-weight:800;color:#143d4d;margin:13px 0 7px;}
.pa-flow-text {font-size:12px;color:#67828e;line-height:1.5;}
.pa-chip {display:inline-block;padding:5px 11px;border-radius:7px;font-weight:700;font-size:12px;margin-right:8px;}
.pa-normal {background:#dff8e9;color:#167d4a;}.pa-bacterial {background:#fff0db;color:#a05b0e;}.pa-viral {background:#ece5ff;color:#6842ad;}
.pa-banner {padding:17px 20px;border-radius:13px;background:#e4f2fd;border:1px solid #d4e7f8;color:#125b8a;font-size:14px;line-height:1.6;}
.pa-warning {padding:17px 20px;border-radius:13px;background:#fff9e4;border:1px solid #f4e7ad;color:#795b16;font-size:14px;line-height:1.6;}
.pa-row {padding:10px 0;border-bottom:1px solid #e6eff1;color:#466877;font-size:14px;}
.pa-row:last-child {border-bottom:none;}
.pa-strong {font-weight:750;color:#103d50;}
div[data-testid="stTabs"] button {font-weight:650;}
@media(max-width:800px){.pa-hero{padding:19px;gap:12px;}.pa-logo{width:58px;height:58px;}.pa-hero-right{display:none;}}
</style>''', unsafe_allow_html=True)

def panel(title, body):
    st.markdown(f'<div class="pa-panel"><h3>{title}</h3>{body}</div>', unsafe_allow_html=True)

def section(title, desc=''):
    st.markdown(f'<div class="pa-section">{title}</div>' + (f'<div class="pa-desc">{desc}</div>' if desc else ''), unsafe_allow_html=True)

def kpi(label, value, note=''):
    st.markdown(f'<div class="pa-kpi"><div class="pa-kpi-label">{label}</div><div class="pa-kpi-value">{value}</div><div class="pa-kpi-note">{note}</div></div>', unsafe_allow_html=True)

st.markdown(f'''<div class="pa-hero">{logo_html}<div><div class="pa-eyebrow">TIBBİ GÖRÜNTÜLEME / ARAŞTIRMA PLATFORMU</div><h1>Pneumonia AI</h1><p class="pa-sub">Akciğer X-Ray Analizi | Proje Hakkında ve Kullanım Rehberi</p></div><div class="pa-hero-right">Derin öğrenme ile pnömoni sınıflandırması, akciğer segmentasyonu ve model değerlendirmesi.</div></div>''', unsafe_allow_html=True)
st.markdown('<div class="pa-banner"><b>Akademik araştırma prototipi.</b> Bu uygulama klinik tanı, tedavi veya hasta triyajı için kullanılmaz. Model skorları kalibre edilmiş klinik olasılıklar değildir.</div>', unsafe_allow_html=True)

tabs = st.tabs(['Proje Hakkında','Veri Seti','Model Mimarileri','Değerlendirme Metrikleri','Sonuçların Yorumlanması','Kullanım Rehberi'])
with tabs[0]:
    section('Platforma Genel Bakış', 'X-Ray görüntülerinden üç sınıflı sınıflandırma, segmentasyon, hata analizi ve model karşılaştırması.')
    a,b,c=st.columns(3)
    with a: panel('🎯 Projenin Amacı','<p>Akciğer röntgenlerinden <b>NORMAL, BACTERIAL ve VIRAL</b> sınıflarını ayırt eden derin öğrenme modellerinin araştırma amaçlı değerlendirilmesi.</p>')
    with b: panel('🫁 Segmentasyon','<p>U-Net V3, görüntüdeki akciğer alanları için tahmini maske üretir. Maske bir uzman referans anotasyonu değildir.</p>')
    with c: panel('📊 Hata Analizi','<p>Yanlış tahminler, yüksek skorlu hatalar, karışıklık matrisleri ve kalibrasyon bulguları ayrıntılı incelenir.</p>')
    section('Proje İş Akışı', 'Kullanıcının görüntü yüklemesinden sonuçların incelenmesine kadar beş temel adım.')
    cols=st.columns(5)
    steps=[('Röntgen Girişi','Akciğer görüntüsü yüklenir.'),('Ön İşleme','Görüntü modele uygun biçime dönüştürülür.'),('Segmentasyon','U-Net V3 akciğer maskesi üretir.'),('Sınıflandırma','Modeller üç sınıfa ait skor üretir.'),('Sonuç ve Analiz','Tahminler, hatalar ve metrikler incelenir.')]
    for i,(title,description) in enumerate(steps):
        with cols[i]:
            st.markdown(f'<div class="pa-flow"><span class="pa-flow-number">{i+1}</span><div class="pa-flow-title">{title}</div><div class="pa-flow-text">{description}</div></div>',unsafe_allow_html=True)
    section('Araştırmanın Kapsamı')
    c1,c2=st.columns(2)
    with c1: panel('Sınıflandırma','<div class="pa-row"><span class="pa-chip pa-normal">NORMAL</span>Pnömoni etiketi bulunmayan görüntüler</div><div class="pa-row"><span class="pa-chip pa-bacterial">BACTERIAL</span>Bakteriyel pnömoni etiketi</div><div class="pa-row"><span class="pa-chip pa-viral">VIRAL</span>Viral pnömoni etiketi</div>')
    with c2: panel('Araştırma Bileşenleri','<div class="pa-row">4 sınıflandırma modeli</div><div class="pa-row">2 ablasyon deneyi</div><div class="pa-row">U-Net V3 akciğer segmentasyonu</div><div class="pa-row">Hata analizi, ROC ve kalibrasyon değerlendirmesi</div>')
    st.markdown('<div class="pa-warning" style="margin-top:18px"><b>Önemli:</b> X-Ray görüntüsünden bakteriyel veya viral etkenin kesin tanısı konulamaz. Sonuçlar yalnızca veri kümesindeki etiketlere ilişkin model tahminleridir.</div>',unsafe_allow_html=True)
with tabs[1]:
    section('Veri Seti','Kaggle Chest X-Ray görüntülerine dayanan, kayıtlı eğitim dağılımı.')
    c=st.columns(4)
    for col,(label,value,note) in zip(c,[('Eğitim toplamı','4.021','Görüntü'),('NORMAL','1.089','%27,1'),('BACTERIAL','1.878','%46,7'),('VIRAL','1.054','%26,2')]):
        with col:kpi(label,value,note)
    st.markdown('')
    st.bar_chart({'NORMAL':1089,'BACTERIAL':1878,'VIRAL':1054},horizontal=True)
    section('Doğrulama ve Test Kümeleri')
    a,b=st.columns(2)
    with a:panel('Ortak doğrulama raporları','<p>Baseline CNN, MobileNetV2 ve Fine-Tuned MobileNetV2 için <b>875 görüntü</b> içeren kayıtlı doğrulama sonuçları.</p>')
    with b:panel('Conservative Group testi','<p>Conservative Group modelinin <b>876 görüntü</b> içeren ayrı retrospektif test kümesi.</p>')
    st.warning('875 görüntülük doğrulama ve 876 görüntülük test kümeleri farklıdır; bu sonuçlardan doğrudan model üstünlüğü çıkarılamaz. Eğitim kümesi için 4.021 sayısı test kümesiyle karıştırılmamalıdır.')
with tabs[2]:
    section('Model Mimarileri','Kullanılan yaklaşımlar ve bunların araştırmadaki rolleri.')
    models=[('Baseline CNN','Konvolüsyon katmanlarıyla görüntü özellikleri öğrenen temel karşılaştırma ağı.'),('MobileNetV2','Hafif CNN mimarisi. Transfer öğrenme ile önceden öğrenilmiş görsel temsillerden yararlanır.'),('Fine-Tuned MobileNetV2','Önceden eğitilmiş katmanların bir bölümünün de veri kümesine uyarlanabildiği model.'),('MobileNetV2 – Conservative Group','Ayrı retrospektif test raporuyla değerlendirilen model. Yöntem ayrıntıları deney kayıtlarından doğrulanmalıdır.'),('U-Net V3','Piksel düzeyinde tahmini akciğer maskesi üretir; sınıflandırıcı değildir.')]
    for i in range(0,len(models),2):
        cols=st.columns(2)
        for col,(name,desc) in zip(cols,models[i:i+2]):
            with col:panel(name,f'<p>{desc}</p>')
    section('Ablasyon Deneyleri')
    st.info('İki ablasyon deneyi, model bileşenleri veya işlem tercihlerinin etkisini incelemek içindir. Deneylerin tam değişkenleri uygulamadaki kayıtlı raporlardan teyit edilmelidir.')
with tabs[3]:
    section('Değerlendirme Metrikleri','Her göstergenin neyi ölçtüğü ve nasıl yorumlanması gerektiği.')
    items=[('Doğruluk (Accuracy)','Doğru tahmin sayısı / toplam örnek sayısı. Sınıf dengesizliğinde tek başına yanıltabilir.'),('Kesinlik (Precision)','Bir sınıf olarak tahmin edilenlerin gerçekten o sınıfa ait olma oranı.'),('Duyarlılık (Recall)','Gerçekte ilgili sınıfa ait örneklerin ne kadarının yakalandığı.'),('F1 Skoru','Precision ve Recall değerlerinin harmonik ortalaması.'),('Makro F1','Her sınıfın F1 değerinin eşit ağırlıklı ortalaması.'),('Dengeli Doğruluk','Sınıf bazlı duyarlılıkların ortalaması.'),('Karmaşıklık Matrisi','Satırlar gerçek, sütunlar tahmin sınıflarıdır. Köşegen doğru, diğer hücreler hatalı tahminleri gösterir.'),('ROC-AUC','Farklı eşiklerde sınıfları ayırabilme yeteneği. %90 AUC, %90 doğruluk değildir.'),('Kesinlik–Duyarlılık Eğrisi','Sınıflandırma eşiğine göre kesinlik ve duyarlılık arasındaki değişim.'),('ECE / Kalibrasyon Hatası','Skor aralıklarında ortalama model skoru ile gözlenen doğruluk farkı. Düşük olması tercih edilir.'),('Brier Skoru','Tahmin skorlarının gerçek etiketlere göre karesel hata ölçüsü. Düşük olması tercih edilir.'),('Yüksek Skorlu Hata','Modelin yüksek skor verdiği ancak yanlış sınıflandırdığı örnekler; klinik güveni göstermez.')]
    for name,desc in items:
        with st.expander(name):st.write(desc)
    st.markdown('<div class="pa-warning"><b>Kalibrasyon notu:</b> Yüksek doğruluk, iyi kalibrasyon anlamına gelmez. Softmax değerleri klinik tanı olasılığı olarak kullanılmamalıdır.</div>',unsafe_allow_html=True)
with tabs[4]:
    section('Kayıtlı Deney Sonuçları','Değerler mevcut rapor ekranlarından aktarılmıştır; burada yeni bir model çalıştırılmamaktadır.')
    st.markdown('**Üç modelin ortak doğrulama sonuçları (n = 875)**')
    st.dataframe({'Model':['Baseline CNN','MobileNetV2','Fine-Tuned MobileNetV2'],'Doğruluk':['%77,71','%77,03','%78,29'],'Makro F1':['%74,93','%77,05','%78,45'],'Yanlış tahmin':[195,201,190]},hide_index=True,use_container_width=True)
    section('Conservative Group | Ayrı Test Kümesi (n = 876)')
    cols=st.columns(4)
    for col,(label,value) in zip(cols,[('Doğruluk','%78,77'),('Makro F1','%78,59'),('Dengeli Doğruluk','%79,25'),('Makro ROC-AUC','%90,71')]):
        with col:kpi(label,value)
    st.warning('Bu test kümesi diğer üç modelin doğrulama kümesiyle aynı değildir; yüzdeleri doğrudan sıralamak metodolojik olarak doğru olmaz.')
    section('Hata Örüntüleri')
    a,b=st.columns(2)
    with a:panel('Baseline CNN','<p>875 görüntünün <b>195</b> tanesi hatalı sınıflandırılmıştır. Karışıklık matrisinde gerçek VIRAL olan <b>113</b> görüntü BACTERIAL tahmin edilmiştir.</p>')
    with b:panel('Conservative Group','<p>876 görüntünün <b>186</b> tanesi hatalıdır. Gerçek BACTERIAL olan <b>89</b> görüntü VIRAL tahmin edilmiştir.</p>')
    st.info('Yüksek skorlu yanlış tahminler özellikle incelenmelidir. Bir örneğe %97 softmax skoru verilmesi, o tahminin klinik olarak %97 doğru olduğu anlamına gelmez.')
with tabs[5]:
    section('Uygulama Kullanım Rehberi','Analiz sayfaları ve sonuçları güvenli biçimde yorumlamak için adımlar.')
    with st.expander('1. Görüntü Analizi',expanded=True):
        st.write('Görüntü Analizi sayfasına geçin, akciğer X-Ray dosyası yükleyin ve analiz düğmesini kullanın. Model skorlarını karşılaştırın; skorları kesin tanı olarak yorumlamayın.')
        st.page_link('app.py',label='Görüntü Analizine Git',icon=':material/home:')
    with st.expander('2. Akciğer Segmentasyonu'):
        st.write('Orijinal X-Ray, tahmini yeşil segmentasyon katmanı ve ikili akciğer maskesini inceleyin. Maske saydamlığı ve görünürlüğü kontrollerini kullanın. Maskeler uzman anotasyonu değildir.')
    with st.expander('3. Model Karşılaştırması ve Ablasyon'):
        st.write('Farklı sınıflandırıcıların çıktılarını ve kayıtlı deneylerini yan yana inceleyin. Aynı veri kümesi ve aynı ölçütler kullanıldığında kıyaslama daha anlamlıdır.')
    with st.expander('4. Hata Analizi'):
        st.write('Model seçin; gerçek/tahmin sınıfı filtrelerini ve skor eşiğini değiştirin. Yanlış sınıflandırılan görüntüleri, karışıklık matrisini, ROC ve kalibrasyon grafiklerini inceleyin.')
        st.page_link('pages/1_Hata_Analizi.py',label='Hata Analizine Git',icon=':material/analytics:')
    with st.expander('5. Sınırlamalar ve araştırma etiği'):
        st.write('Model performansı farklı yaş grupları, hastaneler, cihazlar ve görüntü kalitelerinde değişebilir. Bağımsız dış doğrulama, veri sızıntısı kontrolü ve uzman incelemesi gereklidir.')
    st.error('Bu uygulama tıbbi cihaz veya onaylı klinik karar destek sistemi değildir. Hasta tanısı, triyajı veya tedavisi için kullanmayın.')


# PNEUMONIA_UI_V3: supplementary research documentation
from medical_ui_enhancements import apply_extended_ui as _pa_ext_ui, model_guide as _pa_model_guide
_pa_ext_ui()
st.divider()
_pa_model_guide(["Baseline CNN", "MobileNetV2", "Fine-Tuned MobileNetV2",
                 "MobileNetV2 - Conservative Group", "U-Net V3",
                 "Deney A - Orijinal", "Deney B - Maskeli"],
                title="Model mimarileri: ayrıntılı teknik açıklamalar")
