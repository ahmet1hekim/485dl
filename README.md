# 🎓 Derin Öğrenme ile Yüzden Yaş Tahmini Projesi

Bu proje, **Derin Öğrenme (Deep Learning)** yöntemleri kullanılarak insan yüzlerinden yaş tahmini (Age Regression) yapılması amacıyla geliştirilmiştir.

---

## 1. 📌 Proje Konusu ve Amacı (15 Puan)

### Seçilme Gerekçesi ve Önemi
Yüz analizi, günümüzde biyometrik güvenlikten kişiselleştirilmiş pazarlamaya kadar geniş bir alanda kullanılmaktadır. Yaş tahmini, bu analizin en zorlu problemlerinden biridir çünkü yaşlanma süreci kişiden kişiye (genetik, yaşam tarzı) ve çevresel faktörlere (ışık, makyaj) göre büyük değişiklik gösterir.

### Literatür ve Mevcut Uygulamalar
Geleneksel yöntemlerde (SVM, PCA) el ile çıkarılan özellikler (hand-crafted features) kullanılırken, son yıllarda **Evrişimli Sinir Ağları (CNN)** bu alanda standart haline gelmiştir. Literatürde VGG-Face, ResNet ve EfficientNet mimarileri sıkça kullanılmaktadır. Bu projede, kaynak verimliliği ve başarı oranı dengesi nedeniyle CNN tabanlı modern bir mimari tercih edilmiştir.

---

## 2. 📊 Veri Setinin Belirlenmesi (15 Puan)

### Seçilen Veri Seti: UTKFace
Projede, yaş tahmini literatüründe 'benchmark' olarak kabul edilen **UTKFace** veri seti kullanılmıştır.

* **Çeşitlilik:** 0 ile 116 yaş aralığında, farklı etnik köken ve cinsiyetlerden 20.000'den fazla yüz görüntüsü içerir.
* **Veri İşleme:** Görüntüler modele verilmeden önce 224x224 piksel boyutuna getirilmiş ve normalizasyon (ImageNet standartları: Mean=[0.485...], Std=[0.229...]) işlemine tabi tutulmuştur.
* **Erişim:** Veriler `Hugging Face Datasets` kütüphanesi aracılığıyla dinamik olarak çekilmektedir.

---

## 3. ⚙️ Yöntem ve Algoritma Seçimi (15 Puan)

### Yaklaşım: Transfer Learning (Transfer Öğrenme)
Sıfırdan bir CNN eğitmek yerine, **ImageNet** üzerinde önceden eğitilmiş **ResNet18** mimarisi kullanılmıştır.

### Karşılaştırmalı Analiz ve Seçim Nedeni
1.  **Neden CNN?**: Görüntüden öznitelik çıkarma konusunda MLP veya klasik ML yöntemlerinden çok daha üstündür.
2.  **Neden ResNet18?**:
    * **VGG16'ya göre:** Çok daha az parametre içerir, daha hızlı eğitilir ve 'Vanishing Gradient' problemini 'Skip Connections' ile çözer.
    * **Custom CNN'e göre:** Önceden eğitilmiş ağırlıklar (Weights) sayesinde model, kenar/doku gibi temel özellikleri zaten bilerek başlar. Bu, eğitim süresini saatlerden dakikalara indirir.
3.  **Loss Fonksiyonu:** Yaş tahmini bir regresyon problemidir. Burada MSE (Mean Squared Error) yerine **L1 Loss (Mean Absolute Error)** tercih edilmiştir. Çünkü L1 Loss, veri setindeki yaşlı/genç aykırı değerlere (outliers) karşı daha dayanıklıdır.

---

## 4. 📈 Model Eğitimi ve Değerlendirme (20 Puan)

Model, **PyTorch** kütüphanesi kullanılarak eğitilmiştir.

* **Optimizasyon:** Adam Optimizer kullanılmıştır.
* **Differential Learning Rates:** Modelin ön kısımları (Feature Extractor) bozulmaması için çok düşük öğrenme oranıyla (1e-5), son katman (Classifier) ise hızlı öğrenmesi için yüksek oranla (1e-3) eğitilmiştir.
* **Data Augmentation:** Overfitting'i (aşırı öğrenme) engellemek için eğitim verisine rastgele yatay çevirme, döndürme ve renk değişimleri uygulanmıştır.

**Başarı Kriteri (Metric):** Modelin başarısı MAE (Mean Absolute Error) ile ölçülmüştür. Örn: MAE 5.0 ise, modelin tahmini gerçek yaştan ortalama ±5 yaş sapmaktadır.

---

## 5. 📂 Proje Dokümantasyonu (10 Puan)

Proje dosyaları, yeniden üretilebilirliği sağlamak amacıyla modüler bir yapıda düzenlenmiştir.

```text
.
├── model.py         # ResNet18 mimarisini tanımlayan sınıf
├── train.py         # Veri indirme, ön işleme ve eğitim döngüsü
├── app.py           # Gradio tabanlı web arayüzü ve sunum kodu
├── README.md        # Proje raporu
├── data/            # Eğitim ve test verilerinin tutulduğu klasör
└── model/           # Eğitilmiş model ağırlıkları (.pth)
```

---

## 6. 🎥 Projenin Sunumu ve Çalıştırılması (25 Puan)

Proje, son kullanıcının kolayca test edebilmesi için **Gradio** arayüzü ile sunulmuştur.

### Kurulum
```bash
pip install torch torchvision gradio datasets tqdm pillow matplotlib
```

### Adım 1: Modeli Eğit
```bash
python train.py
```
*(Bu işlem veriyi indirir ve `model/age_resnet.pth` dosyasını oluşturur.)*

### Adım 2: Arayüzü Başlat
```bash
python app.py
```
Tarayıcıda açılan link üzerinden test verisinden rastgele seçilen yüzlerle yaş tahmini yapabilirsiniz.
