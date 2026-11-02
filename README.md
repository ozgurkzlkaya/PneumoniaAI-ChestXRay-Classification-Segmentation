# Pneumonia AI — Araştırma platformu

Üç sınıflı akciğer X-Ray sınıflandırması, U-Net V3 segmentasyonu ve B3 ensemble araştırmasının kaynak kodları.

> **Bu Git deposunda model ağırlıkları, veri kümeleri ve gerçek X-Ray/Grad-CAM görselleri bulunmaz.** Bu çalışma kopyasında 43 toplu deney çıktısı ve 20 örnek düzeyinde CSV korunur. Örnek CSV’ler yalnızca Git index’inden çıkarılmıştır; yerel dosyalar aynı boyut ve SHA-256 değerleriyle korunur. Önceki public Git geçmişi değişmemiştir. Model dosyaları olmadan gerçek sınıflandırma, segmentasyon veya ensemble tahmini yapılamaz.
>
> **Tıbbi kullanım uyarısı:** Akademik araştırma ve eğitim prototipidir. Tanı, tedavi, triyaj veya hasta yönetimi için kullanılmamalıdır. Model skorları kalibre edilmiş klinik hastalık olasılıkları değildir.

## Amaç ve kapsam

Proje; temel CNN ve MobileNetV2 modellerini karşılaştırır, akciğer segmentasyonunun sınıflandırma girdisine etkisini inceler ve kayıtlı deneylerdeki hataları araştırır. Bu ayrı yayın kopyası, çalışan yerel projenin veri ve ağırlık içeren tam dağıtımı değildir.

Uygulama mantığı, eğitim/değerlendirme kodları ve ön işleme hesaplamaları korunmuştur. İki Python dosyasının yalnızca kullanım açıklamasındaki yerel kullanıcı yolu, yayın kopyasında göreli yol örneğiyle değiştirilmiştir. Veri ve model dosyalarının yeniden temini olmadan tüm deneylerin tekrarlanabileceği iddia edilmez. Bazı eski betikler arşivlenmiş veri düzenlerini veya ayrıca sağlanması gereken manifestleri bekler.

## Teknolojiler ve sınıflar

- Python 3.12; yerel referans ortamında Windows/CPU.
- TensorFlow 2.21.0 ve `tf.keras`; Streamlit 1.65.0.
- NumPy, pandas, Pillow ve SciPy.
- scikit-learn, Matplotlib ve Altair; görüntü kalitesi araştırma betiklerinde OpenCV.

Sınıf sırası **NORMAL**, **BACTERIAL**, **VIRAL** olarak sabittir. Arayüzde bunlar Normal, Bakteriyel pnömoni ve Viral pnömoni adlarıyla da gösterilir. Bu adlar veri kümesinin araştırma etiketleridir; klinik kesinlik bildirmez.

[requirements.txt](requirements.txt) uygulama ve araştırma betiklerinin doğrudan bağımlılıklarını birlikte içerir. OpenCV aralığı bu yayın sürümünde `>=4.10,<4.12` olarak sınırlandırılmıştır; böylece NumPy 1.26.2 ile uyumlu 4.10/4.11 sürüm ailesi korunur. Diğer dokuz bağımlılık kısıtı değiştirilmemiştir. Tüm işletim sistemleri ve Python sürümleri için uyumluluk garantisi verilmez.

## Modeller ve nihai yöntem

Dört karşılaştırma modeli korunur: **Baseline CNN**, **MobileNetV2**, **Fine-Tuned MobileNetV2** ve **MobileNetV2 - Conservative Group**. Conservative Group farklı veri bölünmesiyle değerlendirilmiş ayrı bir deneydir.

**U-Net V3 segmentasyonu:** Görüntü 256×256 gri tona ve [0,1] aralığına dönüştürülür. Model çıktısına 0,5 eşik uygulanır; ikili maske orijinal boyuta en yakın komşu yöntemiyle taşınır. Yeşil katman bir görselleştirmedir; saydamlık ayarı model maskesini değiştirmez.

**B3 alpha=0.50:** U-Net V3 maskesi [create_b2_expanded_dataset.py](create_b2_expanded_dataset.py) içindeki `refine` fonksiyonuyla düzeltilir ve genişletilir. Akciğer dışındaki RGB piksel yoğunlukları 0,50 ile çarpılır. Bilgi tamamen sıfırlanmaz. Sınıflandırma girdisi PNG üzerinden TensorFlow bilinear yeniden boyutlandırmayla 224×224 hazırlanır.

**Nihai ensemble:** Orijinal görüntülerle ve B3 görüntülerle eğitilen iki ayrı fine-tuned MobileNetV2 modelinin üç sınıf skoru eşit ağırlıkla birleştirilir:

```text
p_final = 0.50 × p_original + 0.50 × p_B3_alpha050
```

En yüksek birleşik skora sahip sınıf gösterilir. Araştırma kodlarında bulunan bias/karar eşiği taramaları ve iki aşamalı modeller, bu nihai arayüz yöntemiyle aynı değildir.

## Ablasyon ve değerlendirme sınırları

**Deney A** orijinal görüntüyü kullanır. **Deney B** U-Net V3 ikili maskesi dışındaki pikselleri sıfırlar. Bu sert maskeleme, nihai B3 yumuşak maskeleme ve ensemble yöntemiyle aynı değildir. Genişletilmiş maske kullanan `B2_black` da ayrı bir deney koludur.

Uygulama ve rehber kodunda korunan kayıtlı toplu sonuçlar:

| Deney | Değerlendirme | Örnek | Doğruluk | Makro F1 |
| --- | --- | ---: | ---: | ---: |
| Deney A — Orijinal | Doğrulama | 875 | %79,89 | 0,7975 |
| Deney B — Sert maskeli | Doğrulama | 875 | %66,74 | 0,6675 |
| Original + B3 ensemble | Doğrulama | 875 | %80,34 | 0,8049 |
| Original + B3 ensemble | Önceden incelenmiş test | 960 | %80,83 | 0,8010 |

Bu tablo yeni eğitim veya yeni doğrulama sonucu değildir. Mevcut depodaki eski deney raporları korunmuştur. Nihai B3 ensemble doğrulama/test tahmin dosyaları bu klonda bulunmaz; bağımsız yeniden hesaplama için ilgili ağırlıklar ve kayıtlar ayrıca sağlanmalıdır. İki basamaklı gösterimle A/B farkı 13,15 yüzde puandır; ham kayıtlı oranlarla hesaplanan farkın iki basamaklı yuvarlaması 13,14'tür.

Sınıflandırma değerlendirmeleri doğruluk, precision/recall/F1, karışıklık matrisi, ROC/PR eğrileri ve kalibrasyon incelemelerini içerir. Segmentasyon ayrı olarak Dice ve IoU ile ölçülür. Toplu doğrulama metrikleri, tek bir yüklenen görüntünün skoru veya klinik güven derecesi değildir.

Doğrulama kümesi geliştirme sırasında tekrar kullanılmıştır; 960 örnekli test de önceden incelenmiştir. Conservative Group'un 876 örnekli testi ayrı bir değerlendirmedir. Eski üç karşılaştırma modeli için bir kesin kopyanın çıkarıldığı 959 örnekli rapor, nihai ensemble'ın tüm sızıntı riskleri giderilmiş testi olarak sunulamaz.

Özgün bütünlük denetiminde bölünmeler arasında 1 kesin kopya grubu, aynı bölünme içinde 29 kopya grubu, 52 aday kimlik örtüşmesi ve 51 aday görüntü yakalama örtüşmesi raporlanmıştır. Önceki pHash denetimi 104 yakın eşleşme adayı bildirir. Bunlar farklı yöntemlerin bulgularıdır; adaylar doğrulanmış hasta eşleşmeleri değildir ve sayıları birleştirilmez. Hasta düzeyinde bağımsızlık veya bağımsız dış klinik doğrulama kanıtlanmış kabul edilmez.

## Arayüz ve bu dağıtımdaki kullanılabilirlik

| Sayfa | Dosya | Ağırlık/veri/rapor olmadan durum |
| --- | --- | --- |
| Görüntü Analizi | [app.py](app.py) | Giriş, yükleme, önizleme ve temizleme kullanılabilir; gerçek analiz yapılamaz |
| Hata Analizi | [1_Hata_Analizi.py](pages/1_Hata_Analizi.py) | Toplu raporlar okunabilir; örnek CSV’leri olmayan hata tabloları ve tekil görüntü incelemesi kullanılamaz. Nihai ensemble ayrıntıları ayrıca eksiktir |
| Proje Hakkında / Platform Rehberi | [2_Platform_Rehberi.py](pages/2_Platform_Rehberi.py) | Yöntem, kullanım ve kayıtlı sınırlamaları açıklar |

Örnek düzeyindeki 20 CSV olmadan da model ağırlıklarıyla gerçek görüntü analizi ve sonuç indirmeleri çalışır. Hata Analizi’nde yalnızca eksik kayda bağlı hata filtreleri, tekil görüntü incelemesi ve filtrelenmiş hata CSV’si devre dışı kalır. Kayıtlı toplu karışıklık matrisi/kalibrasyon dosyaları ve Conservative Group raporları mevcutsa bu bölümler bağımsız gösterilir. Toplu kayıtlar da eksikse ilgili bölüm bilgi verir; diğer bölümler çalışmaya devam eder.

Model dosyaları sağlandığında Görüntü Analizi'nin dört sonuç sekmesi **Nihai Tahmin**, **Segmentasyon**, **Diğer Modeller** ve **Ablasyon Deneyleri**'dir. Üç PNG ve iki CSV indirmesi analiz sonuçlarından üretilir; bu dağıtımda ağırlıklar olmadığı için gerçek sonuç indirmeleri oluşturulamaz. Oturum ve SHA-256 kontrol mantığı korunmuştur.

Ortak görünüm [medical_theme.py](medical_theme.py), [medical_ui_enhancements.py](medical_ui_enhancements.py) ve [.streamlit/config.toml](.streamlit/config.toml) dosyalarındadır. Tema yapılandırması gizli bilgi içermeden kopyalanmıştır.

## Proje yapısı

```text
PneumoniaAI-ChestXRay-Classification-Segmentation/
├── app.py
├── medical_theme.py
├── medical_ui_enhancements.py
├── create_b2_expanded_dataset.py
├── evaluate_ensemble.py
├── pages/
│   ├── 1_Hata_Analizi.py
│   └── 2_Platform_Rehberi.py
├── scripts/
│   ├── analysis/
│   ├── comparison/
│   ├── diagnostics/
│   ├── evaluation/
│   ├── preprocessing/
│   ├── training/
│   ├── visualization/
│   └── generate_research_figures.py
├── experiments/
│   ├── classification_ablation/
│   ├── classification_error_analysis/
│   └── conservative_group_training/
├── PneumoniaAI_Training_Source/
├── b3_preprocessing_source/
├── assets/pneumonia_ai_logo.png
├── .streamlit/config.toml
├── README.md
├── requirements.txt
└── .gitignore
```

[scripts/](scripts/) güncel eğitim, analiz, değerlendirme, ön işleme ve tanılama kodlarını içerir. [PneumoniaAI_Training_Source/](PneumoniaAI_Training_Source/) ve [b3_preprocessing_source/](b3_preprocessing_source/) ek küçük kaynak kodlarını korur. Betikleri çalıştırmadan önce bekledikleri girdiler ve üretecekleri çıktı yolları kontrol edilmelidir; hepsi yalnızca okuma yapan araçlar değildir.

## Windows PowerShell kurulumu ve çalıştırma

Depoyu yeni bir çalışma klasörüne almak için:

```powershell
git clone https://github.com/ozgurkzlkaya/PneumoniaAI-ChestXRay-Classification-Segmentation.git
Set-Location ".\PneumoniaAI-ChestXRay-Classification-Segmentation"
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mevcut çalışma kopyasındaysanız klonlama ve klasör değiştirme adımlarını atlayın. `python --version` çıktısı 3.12.x olmalıdır; farklıysa Python 3.12 yorumlayıcısının tam yolunu kullanın. Doğrudan sanal ortam yorumlayıcısını çağırmak, `Activate.ps1` veya PowerShell yürütme politikası değişikliği gerektirmez. Varsayılan adres `http://localhost:8501`'dir; port doluysa çalıştırma komutuna `--server.port 8513` eklenebilir.

Windows/Python 3.12.2 üzerinde bağımsız sanal ortam kurulumu, pip check ve sekiz yerel ağırlıkla gerçek analiz daha önce doğrulanmıştır. Büyük paketler yerel önbellekten, eksik altı wheel resmi PyPI’dan sağlanmıştır; test başka işletim sistemleri için garanti değildir. Başka projelerin global paket çakışmalarını taşımamak için ayrı ortam önerilir.

Bu güncelleme henüz yalnızca yerel feature dalındadır; mevcut public main’i klonlamak bu değişiklikleri getirmez. Güncelleme yayımlandığında kurulumdan önce doğru dal/tag seçilmelidir.

## Model ve veri dosyalarının ayrıca temini

**Model ağırlıkları ve veri kümeleri Git çalışma ağacına dahil değildir.** Mevcut [Pneumonia AI v1.0.0 — Model Weights](https://github.com/ozgurkzlkaya/PneumoniaAI-ChestXRay-Classification-Segmentation/releases/tag/v1.0.0) korunmaktadır. Bu Release, 11 Keras modeli ve özgün göreli yollarla SHA-256 özetlerini içeren [model_manifest.csv](https://github.com/ozgurkzlkaya/PneumoniaAI-ChestXRay-Classification-Segmentation/releases/download/v1.0.0/model_manifest.csv) dosyasını içerir.

Uygulama modelleri otomatik indirmez; Release ekleri elle indirilip aşağıdaki yerel yollara yerleştirilmelidir. Original kolu ile Deney A aynı checkpoint’i kullandığından benzersiz ağırlık sayısı sekizdir.

Yeni uygulamanın sekiz ayrı ağırlık gereksiniminin yedisi bu Release ile karşılanır. Release adları çakışmaları önlemek için farklı olabilir; manifestteki özgün dosya adı ve göreli klasör geri yüklenmelidir. Dosya adının yanında SHA-256, model sürümü, sınıf sırası ve ön işleme uyumu kontrol edilmelidir.

| Kullanım | Uygulamanın beklediği göreli yol | v1.0.0 eki |
| --- | --- | --- |
| Baseline CNN | `best_pneumonia_model.keras` | `best_pneumonia_model.keras` |
| MobileNetV2 | `best_transfer_model.keras` | `best_transfer_model.keras` |
| Fine-Tuned MobileNetV2 | `best_finetuned_model.keras` | `best_finetuned_model.keras` |
| U-Net V3 | `best_segmentation_model_v3.keras` | `best_segmentation_model_v3.keras` |
| Conservative Group | `experiments/conservative_group_training/control/best_finetuned.keras` | `conservative_control_finetuned.keras` |
| Original kolu ve Deney A | `experiments/classification_ablation/original/best_finetuned.keras` | `ablation_original_finetuned.keras` |
| Deney B | `experiments/classification_ablation/masked/best_finetuned.keras` | `ablation_masked_finetuned.keras` |
| B3 alpha=0.50 kolu | `experiments/classification_ablation/b3_alpha050/best_finetuned.keras` | **Bu Release'te yok** |

**B3 alpha=0.50 ağırlığı v1.0.0'da bulunmaz.** Dolayısıyla v1.0.0 tek başına yeni ensemble uygulamasını tamamlamaz. Sert maskeli Deney B modeli, B3 ağırlığının yerine kullanılamaz. Eksik ağırlığın temin/dağıtım yöntemi ayrıca kararlaştırılmalıdır; burada yeni bir model Release'i veya doğrulanmamış indirme bağlantısı oluşturulmamıştır.

Eğitim için sınıflandırma verileri `dataset/{train,validation,test}/{NORMAL,BACTERIAL,VIRAL}/` düzeninde ayrıca gerekir. Segmentasyon eğitimi ilgili görüntü/maskeleri ve kodun beklediği test bölünmesi kayıtlarını gerektirir; örnek tanımlayıcıları içeren `segmentation_test_files.txt` bu çalışma kopyasında yoktur.

Mevcut `experiments/classification_error_analysis/validation/` ve Conservative Group raporları korunmuştur. Eski `ensemble_analysis` kayıtları nihai Original + B3 değerlendirmesinin yerine geçmez. Nihai modelin ayrıntılı hata incelemesi için gereken `results/classification/ablation_uniform_evaluation/ensemble_50_50/ensemble_predictions.csv` ve `results/classification/fixed_ensemble_test_evaluation/test_predictions.csv` bu klonda bulunmaz. Bunlar haricî girdilerin beklenen konumlarıdır; eksik dosyalara verilmiş bağlantılar değildir.

## Yayın kapsamı ve mevcut raporlar

[.gitignore](.gitignore), seçilmiş kaynak kodlarını, belgeleri, tema dosyasını, logoyu ve yeniden incelenen 43 mevcut bilimsel çıktıyı tam dosya yollarıyla izin listesine alır. Yeni veri, ağırlık, ZIP, tıbbi görüntü veya incelenmemiş rapor varsayılan olarak kapsam dışındadır. Yayın planındaki kesin 20 örnek CSV ayrıca tam yollarıyla dışlanmıştır; bu kurallar mevcut Git takibini kendiliğinden değiştirmez.

Mevcut 63 deney dosyasının tamamı içerikleri değiştirilmeden korunmuştur:

| Kategori | Dosya | Yerel entegrasyondaki durum |
| --- | ---: | --- |
| Toplu metrik, eğitim geçmişi ve boş eksik-görüntü kaydı | 36 | Açık örnek tanımlayıcısı, mutlak kullanıcı yolu ve anahtar/parola imzası taraması temiz |
| Karışıklık matrisi, ROC ve PR grafiklerinin PNG/PDF sürümleri | 6 | Toplu grafik; gerçek X-Ray görseli değil |
| Conservative Group sayısal olasılık dizileri | 1 | 876 etiket ve 876×3 sayısal skor; nesne dizisi veya açık örnek kimliği yok |
| Örnek düzeyinde tanımlayıcı içeren CSV'ler | 20 | **Yerelde korunuyor; sonraki kaynak sürümlerinin Git ağacına dahil değil** |

**Önemli:** `.gitignore`, zaten takip edilen dosyaları takipten çıkarmaz. Son gruptaki 20 CSV yalnızca Git index’inden çıkarılmıştır; yerel dosyalar silinmemiş veya anonimleştirilmemiştir, boyutları ve SHA-256 değerleri korunmuştur. Veri kümesi örnek/grup tanımlayıcıları, etiketler ve görüntü başına tahminler birlikte bulunmaktadır. Bunların gerçek hasta kimliği olduğu iddia edilmez; yeniden paylaşım izni ve bağlantı kurulabilirliği dosya bazında değerlendirilmelidir. Kullanıcı kararı doğrultusunda mevcut public geçmiş korunacak; bu dosyalar sonraki kaynak sürümlerinin Git ağacına alınmayacaktır. Bu kayıtlar mevcut public main ve v1.0.0 geçmişinde de bulunmaktadır; index dışlaması sonraki commit’lerin dosya ağacını etkiler ve eski geçmişteki erişimi kaldırmaz.

Ana proje ve önceden hazırlanmış yayın kopyası değiştirilmez. Model, veri kümesi, gerçek X-Ray/Grad-CAM görselleri, yerel yedekler ve arşivler çalışma ağacına eklenmemiştir. Mevcut bilimsel metrikler veya sonuç dosyaları değiştirilmemiştir.

Tıbbi kullanım için doğrulanmış bir sistem veya haricî literatür üstünlüğü iddiası yoktur. Akademik kullanımda veri bölünmesi, ağırlık sürümü, ön işleme ve ilgili deney kaydı birlikte belirtilmelidir. Üçüncü taraf veri/model lisansları ayrıca kontrol edilmelidir; bu çalışma kopyası yeni bir lisans veya paylaşım hakkı tanımlamaz.
