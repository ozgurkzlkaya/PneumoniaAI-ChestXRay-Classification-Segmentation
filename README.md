# Pneumonia AI — Chest X-Ray Classification & Lung Segmentation

**Üç sınıflı pnömoni sınıflandırması · Akciğer segmentasyonu · Ablasyon deneyleri · Model hata analizi · Streamlit**

Pneumonia AI, pediatrik akciğer röntgenlerini **NORMAL**, **BACTERIAL** ve **VIRAL** olarak sınıflandıran; U-Net ile akciğer bölgelerini segmentlere ayıran ve model sonuçlarını karşılaştırmalı olarak incelemeyi sağlayan akademik bir medikal görüntü işleme projesidir.

> [!IMPORTANT]
> **Yalnızca araştırma ve eğitim amaçlıdır.** Klinik kullanım için doğrulanmış bir tıbbi cihaz değildir. Model çıktıları tanı, tedavi veya klinik karar amacıyla kullanılmamalıdır. Gösterilen model skorları kalibre edilmiş klinik hastalık olasılıkları değildir.

## İçindekiler

- [Projenin kapsamı](#projenin-kapsamı)
- [Kullanılan veri setleri](#kullanılan-veri-setleri)
- [Sınıflandırma modelleri ve sonuçları](#sınıflandırma-modelleri-ve-sonuçları)
- [Akciğer segmentasyonu ve Dice sonuçları](#akciğer-segmentasyonu-ve-dice-sonuçları)
- [Ablasyon deneyleri](#ablasyon-deneyleri)
- [Hata analizi](#hata-analizi)
- [Streamlit uygulaması](#streamlit-uygulaması)
- [Kurulum ve çalıştırma](#kurulum-ve-çalıştırma)
- [Proje yapısı](#proje-yapısı)
- [Bilimsel sınırlamalar](#bilimsel-sınırlamalar)
- [Veri ve model dağıtımı](#veri-ve-model-dağıtımı)

## Projenin kapsamı

1. **Pnömoni sınıflandırması:** Baseline CNN, MobileNetV2 ve Fine-Tuned MobileNetV2 yaklaşımlarının karşılaştırılması.
2. **Ek eğitim yaklaşımı:** Conservative Group MobileNetV2 modelinin ayrı veri bölünmesi kapsamında incelenmesi.
3. **Akciğer segmentasyonu:** U-Net V1, V2 ve V3 mimarilerinin referans akciğer maskelerine göre değerlendirilmesi.
4. **Ablasyon:** Orijinal X-Ray görüntüleri ile segmentasyon maskesi uygulanmış görüntüler üzerinden sınıflandırma deneyleri.
5. **Hata analizi:** Yanlış tahminler, karışıklık matrisleri, sınıf bazlı metrikler, kalibrasyon incelemesi ve modeller arası ortak hatalar.
6. **Web arayüzü:** Streamlit ile görüntü analizi, deney sonuçları ve proje rehberi.

### Teknolojiler

| Alan | Kullanılan araçlar |
|---|---|
| Dil | Python 3.12 |
| Derin öğrenme | TensorFlow / Keras |
| Sınıflandırma | CNN, MobileNetV2, transfer learning, fine-tuning |
| Segmentasyon | U-Net |
| Arayüz | Streamlit |
| Veri işleme | NumPy, Pandas, SciPy, Pillow |
| Değerlendirme | Scikit-learn |
| Görselleştirme | Matplotlib, Altair |

## Kullanılan veri setleri

Bu çalışmada **pnömoni sınıflandırması** ile **akciğer segmentasyonu** için farklı veri kaynakları kullanılmıştır. Bu görevlerin sonuçları aynı metrikle ölçülmez.

### 1. Pediatrik Chest X-Ray — pnömoni sınıflandırması

Üç sınıflı sınıflandırma çalışmasında pediatrik Chest X-Ray görüntüleri kullanılmıştır.

| Sınıf | Görüntü sayısı |
|---|---:|
| NORMAL | 1.583 |
| BACTERIAL | 2.780 |
| VIRAL | 1.493 |
| **Toplam** | **5.856** |

| Veri bölümü | Görüntü sayısı |
|---|---:|
| Eğitim | 4.021 |
| Doğrulama | 875 |
| Test | 960 |
| **Toplam** | **5.856** |

**Not:** Bu üç sınıflı ayrım, projede kullanılan etiketleme düzenidir. Bakteriyel ve viral pnömoninin yalnızca X-Ray görüntüsüyle klinik olarak kesin ayrılabileceği anlamına gelmez.

### 2. Montgomery County Chest X-Ray — akciğer segmentasyonu

Montgomery görüntüleri ve referans akciğer maskeleri, segmentasyon çıktılarının değerlendirilmesinde kullanılmıştır. Aşağıdaki Dice değerleri, tahmin edilen maskeler ile referans maskeler arasındaki örtüşmeyi ifade eder.

### 3. Shenzhen Chest X-Ray — akciğer segmentasyonu

Shenzhen görüntüleri ve referans akciğer maskeleri, segmentasyonun farklı bir görüntü kümesindeki performansını değerlendirmek için kullanılmıştır.

**Kaynak ve protokol notu:** Veri setlerinin tam indirme bağlantıları, lisansları, kullanılan segmentasyon görüntü sayıları ve eğitim/doğrulama/test ayrımlarının ayrıntıları bu teslim özetinden doğrulanamadığı için burada kesin sayı veya bağlantı verilmemiştir. Bunlar deney kayıtlarından teyit edilerek eklenmelidir. Veri setlerinin adları tek başına aynı değerlendirme protokolünü garanti etmez.

## Sınıflandırma modelleri ve sonuçları

### Modeller

- **Baseline CNN:** Sıfırdan oluşturulan evrişimli sinir ağı.
- **MobileNetV2:** Önceden eğitilmiş ağ üzerinden transfer öğrenme.
- **Fine-Tuned MobileNetV2:** MobileNetV2'nin belirli katmanlarının yeniden eğitilmesi.
- **Conservative Group MobileNetV2:** Ayrı veri bölünmesi / eğitim düzeninde değerlendirilen ek model.

### 960 görüntülük test kümesi

| Model | Accuracy | Macro F1 | Viral Recall |
|---|---:|---:|---:|
| Baseline CNN | %81,56 | 0,7726 | %41,18 |
| MobileNetV2 | %80,83 | 0,7868 | %57,01 |
| **Fine-Tuned MobileNetV2** | **%81,98** | **0,8003** | **%58,82** |

Bu üç model arasında **Fine-Tuned MobileNetV2**, raporlanan test kümesinde en yüksek Accuracy ve Macro F1 değerlerini elde etmiştir. Bununla birlikte Viral Recall değerinin %58,82 olması, viral sınıfın önemli bir bölümünün hâlâ kaçırıldığını gösterir.

> **Karşılaştırılabilirlik:** Conservative Group modeli farklı veri bölünmesinde değerlendirilmiştir. Bu nedenle sonuçları yukarıdaki ortak test tablosuna doğrudan eklenmemiştir. Ayrıca uygulamadaki **875 görüntülük doğrulama analizleri**, buradaki **960 görüntülük test sonuçlarından** ayrı tutulmalıdır.

## Akciğer segmentasyonu ve Dice sonuçları

Akciğer bölgelerini belirlemek için **U-Net V1, V2 ve V3** değerlendirilmiştir.

| Model | Montgomery Dice | Shenzhen Dice |
|---|---:|---:|
| U-Net V1 | 0,9655 (**%96,55**) | 0,8810 (**%88,10**) |
| U-Net V2 | 0,7617 (**%76,17**) | 0,9483 (**%94,83**) |
| **U-Net V3** | **0,9719 (%97,19)** | **0,9500 (%95,00)** |

### Referans maskelere ne kadar yakınız?

**Dice katsayısı**, modelin ürettiği akciğer maskesinin veri setindeki **referans (ground truth) maske** ile ne ölçüde örtüştüğünü gösterir:

\[
\mathrm{Dice}(A,B)=\frac{2|A\cap B|}{|A|+|B|}
\]

- **Montgomery:** U-Net V3 Dice = **0,9719** → **%97,19 örtüşme skoru**.
- **Shenzhen:** U-Net V3 Dice = **0,9500** → **%95,00 örtüşme skoru**.
- İki raporlanan Dice skoru arasındaki fark **2,19 yüzde puandır**.

Bu yüzdeler **piksel doğruluğu (pixel accuracy)** veya **görüntülerin yüzde kaçının kusursuz segmentlendiği** anlamına gelmez. Ayrıca veri setinin kendisine ait bir “başarı yüzdesi” değildir; **bizim modelimizin referans maskelere göre ölçülen Dice performansıdır**.

### Kendi modellerimiz arasındaki karşılaştırma

- Montgomery üzerinde V3, V1'e göre **0,64 yüzde puan** daha yüksek Dice elde etmiştir (%97,19 − %96,55).
- Shenzhen üzerinde V3, V2'ye göre **0,17 yüzde puan** daha yüksek Dice elde etmiştir (%95,00 − %94,83).
- Bu değerler yalnızca raporlanan deney sonuçlarının karşılaştırmasıdır; istatistiksel anlamlılık veya haricî klinik genelleme kanıtı değildir.

### Literatürdeki yöntemlere göre durumumuz

Bu README'de **aynı görüntüler, aynı referans maskeleri, aynı veri bölünmesi ve aynı Dice hesaplama protokolü** ile doğrulanmış bir haricî makale karşılaştırması bulunmamaktadır. Bu nedenle “literatürdeki modelden %X daha başarılıyız” iddiasında bulunulmuyor. Böyle bir kıyaslama, karşılaştırılan çalışmanın tam protokolü doğrulandıktan sonra eklenmelidir.

## Ablasyon deneyleri

Akciğer segmentasyonunun sınıflandırmaya etkisini incelemek amacıyla iki koşul kullanılmıştır:

| Deney | Girdi | Amaç |
|---|---|---|
| A — Original | Orijinal Chest X-Ray | Tam görüntü üzerinden sınıflandırma |
| B — Masked | Akciğer maskesi uygulanmış Chest X-Ray | Akciğer dışı bölgelerin etkisini inceleme |

Deney dosyaları `experiments/classification_ablation/` altında yer alır. Deneyin eğitim ve değerlendirme koşulları karşılaştırılırken aynı bölünme ve metrik tanımları dikkate alınmalıdır.

## Hata analizi

`pages/1_Hata_Analizi.py` sayfası, `experiments/classification_error_analysis/validation/` klasöründeki raporlardan yararlanır.

İncelenen başlıklar:

- Model bazlı yanlış sınıflandırmalar ve örnek görüntüler
- Karışıklık matrisleri
- Sınıf bazlı metrikler
- Model skorlarının kalibrasyon analizi
- Ortak yanlış tahminler ve model uyuşmazlıkları
- Bakteriyel / viral sınıflar arasındaki karışıklıklar
- Conservative Group deneyine ait ayrı değerlendirme çıktıları

**Not:** Bu sayfadaki ortak doğrulama kümesi **875 görüntüdür**; yukarıdaki 960 görüntülük test kümesiyle karıştırılmamalıdır.

## Streamlit uygulaması

| Sayfa | İçerik |
|---|---|
| Ana uygulama (`app.py`) | X-Ray yükleme, sınıflandırma, model karşılaştırması, U-Net V3 segmentasyonu ve deney incelemeleri |
| `pages/1_Hata_Analizi.py` | Yanlış tahminler, karışıklık matrisleri, ortak hatalar ve analiz raporları |
| `pages/2_Platform_Rehberi.py` | Uygulamanın kullanımı ve yöntemlerin açıklanması |

### Örnek kullanım akışı

1. Uygulamayı başlatın.
2. Desteklenen formatta bir akciğer X-Ray görüntüsü yükleyin.
3. İlgili sınıflandırma modelinin skorlarını ve tahminini inceleyin.
4. U-Net V3 ile üretilen akciğer maskesini görüntüleyin.
5. Hata Analizi sayfasından modellerin raporlanan performanslarını karşılaştırın.

## Kurulum ve çalıştırma

### Gereksinimler

- **Python 3.12** önerilir.
- `requirements.txt` içindeki paketler.
- Tam işlevsellik için ilgili **`.keras` model ağırlıkları**.
- Hata analizi görüntülerini göstermek için beklenen klasör yapısındaki **doğrulama görüntüleri**.

```bash
# Depoyu indirin
git clone https://github.com/ozgurkzlkaya/PneumoniaAI-ChestXRay-Classification-Segmentation.git
cd PneumoniaAI-ChestXRay-Classification-Segmentation

# Sanal ortam oluşturun
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Linux / macOS:**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Uygulama genellikle **http://localhost:8501** adresinde açılır.

> [!WARNING]
> **Kaynak kod deposu tek başına tam çalışır uygulama paketi olmayabilir.** Büyük model dosyaları ve görüntüler GitHub'daki ilk kod yüklemesine dahil edilmiyorsa model çıkarımı veya görüntü bazlı hata analizi eksik kalabilir. Bu dosyaların dağıtım bağlantıları yayımlandığında aşağıdaki bölüme eklenmelidir. Farklı bilgisayarda temiz ortam kurulumu henüz ayrıca doğrulanmamıştır.

## Proje yapısı

Aşağıdaki ağaç, uygulama teslim klasörünün **temsili** yapısını gösterir; GitHub'a ilk yüklemede model ve veri dosyaları hariç tutulabilir.

```text
PneumoniaAI_Uygulama/
├── app.py
├── medical_theme.py
├── medical_ui_enhancements.py
├── pages/
│   ├── 1_Hata_Analizi.py
│   └── 2_Platform_Rehberi.py
├── assets/
│   └── pneumonia_ai_logo.png
├── experiments/
│   ├── classification_ablation/
│   ├── classification_error_analysis/
│   │   └── validation/
│   └── conservative_group_training/
├── dataset/                       # Teslim paketinde 875 görüntü
├── best_pneumonia_model.keras     # GitHub kod deposunda olmayabilir
├── best_transfer_model.keras
├── best_finetuned_model.keras
├── best_segmentation_model_v3.keras
├── requirements.txt
├── README.md
└── .gitignore
```

Ek deney modelleri `experiments/` altındaki ilgili koşul klasörlerinde bulunur.

## Veri ve model dağıtımı

**Akademik uygulama teslim ZIP'i** ile **GitHub kaynak kod deposu** farklı amaçlara hizmet eder:

- **Teslim ZIP'i:** Model ağırlıklarını, uygulama dosyalarını, gerekli deney raporlarını ve 875 doğrulama görüntüsünü içerir.
- **GitHub kaynak kod deposu:** Kod, açıklamalar ve uygun boyuttaki deney çıktıları için hazırlanır. Büyük `.keras` dosyaları ve görüntü verileri ilk aşamada hariç tutulabilir.

**Model indirme bağlantısı:** Henüz yayımlanmadı.

**Veri erişimi:** Veri setlerinin resmî kaynaklarından, kullanım koşulları ve lisansları doğrulanarak edinilmelidir. Veri görüntülerini izinsiz yeniden dağıtmayın; hasta bilgisi içeren dosyaları yayımlamayın.

## Bilimsel sınırlamalar

1. Veri setleri, tüm yaş gruplarını, cihazları veya klinik ortamları temsil etmeyebilir.
2. **NORMAL / BACTERIAL / VIRAL** etiketleri veri seti etiketleridir; klinik tanı yerine geçmez.
3. Test, doğrulama ve Conservative Group deneyleri farklı değerlendirme koşullarına sahip olabilir; metrikler bağlamından koparılarak karşılaştırılmamalıdır.
4. Segmentasyon sonuçları **Montgomery ve Shenzhen** referans maskelerine göredir; pediatrik pnömoni görüntülerinde bağımsız segmentasyon doğrulaması yapılmış olduğu iddia edilmez.
5. Dice skoru, klinik başarı oranı veya piksel accuracy değildir.
6. Haricî literatürle adil kıyaslama için eşleşen değerlendirme protokolü gerekir.
7. Model skorları klinik olasılık olarak yorumlanmamalıdır.

## Proje durumu

**Akademik geliştirme ve uygulama teslim paketi tamamlandı.** Kaynak kodunun GitHub üzerinden yayımlanması ve model/veri dağıtımının belgelenmesi ayrı aşamalardır.

---

*Pneumonia AI — Medical Image Processing / Deep Learning Academic Project*
