--- 
layout: post
title: "Keras API'leri"
subtitle: "Keras'ın dökümantasyonundan özetlenmiş API rehberi"
cover-img: /assets/img/keras-cover.webp
thumbnail-img: /assets/img/keras-thumb.webp
share-img: /assets/img/keras-thumb.webp
tags: [makine öğrenmesi, keras, yazılım, python, ai, derin öğrenme]
author: Tunahan Yardımcı
date: 2025-03-1 15:55
---

# Giriş

Bu dosya, Keras API’lerini yeni başlayanlar için son derece detaylı ve açıklayıcı bir şekilde ele almaktadır.
Her bir terim, fonksiyon ve konsept, basitten ileri düzeye kadar açıklanacak ve örneklerle desteklenecektir.

Keras, derin öğrenme modelleri oluşturmak için kullanılan yüksek seviyeli bir API'dir.
TensorFlow gibi arka uçlarla (backend) çalışarak sinir ağları, optimizasyon algoritmaları ve veri işleme fonksiyonları sunar.
Bu rehberde; model tanımlama, eğitim, katmanlar, callback’ler, optimizasyon yöntemleri, metrikler, veri yükleme, 
karışık hassasiyet (mixed precision) ve daha pek çok konu detaylı olarak ele alınacaktır.
---
# 1. Model ve Sequential Sınıfları

## 1.1 Model Sınıfı

Model sınıfı, karmaşık sinir ağı mimarileri oluşturmak için temel yapı taşıdır.
Bu sınıf, farklı katmanların birbirine bağlanarak tek bir bütün model oluşturmasını sağlar.

Örneğin:
- **ResNet (Residual Network):** 
  - ResNet, derin ağlarda yaşanan gradyan kaybolması sorununu aşmak için “atlama bağlantıları” (skip connections) kullanır.
  - Bu yapı, çok katmanlı ağlarda öğrenmenin daha verimli gerçekleşmesini sağlar.
  - Model sınıfı sayesinde, ResNet mimarisi gibi karmaşık yapılar kolayca inşa edilebilir.

**Önemli Metotlar:**
- `summary()`: 
  - Modelin mimarisini, katman sayısını ve her katmandaki parametreleri tablo şeklinde özetler.
  - Bu metot, modelin genel yapısını anlamak için son derece faydalıdır.
- `get_layer()`: 
  - Model içerisindeki belirli bir katmanı, ismi veya index numarası ile çağırmanızı sağlar.
  - Bu yöntem, model üzerinde detaylı inceleme yapmak istediğinizde kullanışlıdır.

## 1.2 Sequential Sınıfı

Sequential sınıfı, katmanların birbirine sıralı bir şekilde eklenmesini sağlayan basit bir model yapısıdır.

**Nasıl Çalışır?**
- Her eklenen katman, bir önceki katmandan gelen çıktıyı alır ve bu çıktıyı işleyerek sonraki katmana aktarır.
- Bu yapı, özellikle basit sinir ağı modelleri için idealdir.

**Örnek Kullanım:**
- Giriş katmanı, bir veya daha fazla gizli katman ve çıkış katmanı ekleyerek basit bir ileri yayılım (feedforward) ağı oluşturabilirsiniz.
- Böylece, temel sınıflandırma veya regresyon problemleri kolayca çözülebilir.

####################################################################################################
# 2. Model Training API

Model Training API, modelinizin nasıl eğitileceğini, doğrulanacağını ve test edileceğini belirler.
Bu bölüm, modelin öğrenme sürecini yönetmek için gerekli tüm ayarları içerir.

## 2.1 Eğitim ve Değerlendirme Süreci

Modeli eğitirken, veri seti üzerinde yapılan her iterasyonda ağırlıklar güncellenir.
Bu süreçte kullanılan parametreler şunlardır:
- **Optimizer (Optimizasyon Algoritması):** 
  - Modelin ağırlıklarını güncellemek için kullanılır.
  - Hangi algoritmanın kullanılacağı, modelin öğrenme hızını ve stabilitesini etkiler.
- **Loss (Kayıp) Fonksiyonu:** 
  - Modelin tahminleri ile gerçek değerler arasındaki farkı hesaplar.
  - Bu fark, modelin “ne kadar yanlış” olduğunu belirler.
- **Metrikler (Metrics):** 
  - Modelin performansını ölçmek için kullanılır.
  - Örneğin, doğruluk (accuracy), kayıp (loss) gibi metrikler izlenebilir.

**Önemli Metot:**
- `compile()`: 
  - Modelin hangi optimizer, loss fonksiyonu ve metriklerle eğitileceğini belirler.
  - Bu metot, modelin eğitim sürecinin başlamasından önce mutlaka çağrılmalıdır.

####################################################################################################
# 3. Saving (Kaydetme) API

Eğitim tamamlandıktan sonra, modelin ağırlıklarını ve yapısını kaydetmek önemlidir.
Bu API sayesinde, eğitilmiş modeller daha sonra tekrar yüklenebilir ve kullanılabilir.

**Kullanım Senaryoları:**
- Uzun süren eğitimlerin ardından, modelin bir kopyasını saklamak.
- Model üzerinde denemeler yaparken, en iyi performans gösteren hali kaydedilerek geri yüklenmesi.
- Üretim ortamına geçişte, modeli yeniden eğitmeden doğrudan kullanılabilir hale getirmek.

####################################################################################################
# 4. Layers (Katmanlar) API

Keras’ın temel taşlarından biri olan katmanlar, modelin yapı taşlarını oluşturur.
Her bir katman, veriyi alır, işler ve bir sonraki katmana aktarır.

## 4.1 Base Layer Class

- **Açıklama:**
  - Tüm katmanlar, ortak fonksiyonelliklerini sağlayan Base Layer sınıfından türetilir.
  - İleri yayılım (forward propagation), ağırlıkların yönetimi ve diğer ortak işlemler burada tanımlıdır.

## 4.2 Layer Activations

- **Açıklama:**
  - Aktivasyon fonksiyonları, bir katmandan çıkan veriyi işleyerek nöronların ateşlenme kararını verir.
- **Örnekler:**
  - **ReLU (Rectified Linear Unit):** Negatif değerleri sıfıra indirger.
  - **Sigmoid:** Çıktıyı 0 ile 1 arasında sınırlar.
  - **Tanh:** Çıktıyı -1 ile 1 arasında sıkıştırır.

## 4.3 Layer Weight Initializers

- **Açıklama:**
  - Katmanlardaki ağırlıkların ilk değerlerinin belirlenmesinde kullanılır.
  - Rastgele veya belirli dağılımlara göre ağırlıkları başlatır.
- **Örnek:**
  - **Glorot Uniform:** Ağırlıkları belirli bir aralıkta rastgele seçer.

## 4.4 Layer Weight Regularizers

- **Açıklama:**
  - Ağırlıkların aşırı büyümesini önleyerek overfitting (aşırı öğrenme) riskini azaltır.
- **Örnekler:**
  - **L1 Regularization:** Ağırlıkların mutlak değerlerine ceza uygular.
  - **L2 Regularization:** Ağırlıkların karesine dayalı ceza ekler.

## 4.5 Layer Weight Constraints

- **Açıklama:**
  - Ağırlıkların belirli sınırları aşmasını engelleyen kısıtlamalar getirir.
  - Bu, modelin daha dengeli ve stabil öğrenmesini sağlar.

## 4.6 Core Layers

- **Açıklama:**
  - Sinir ağlarında en temel ve en çok kullanılan katmanlardır.
- **Alt Kategoriler:**
  - **Convolutional (Conv) Layers:**
    - Görüntü işleme ve diğer uzaysal verilerde, filtreleme işlemleri yapar.
    - Örneğin, bir görüntüdeki kenarları veya dokuları algılamaya yarar.
  - **Pooling Layers:**
    - Uzaysal boyutları (örneğin genişlik ve yükseklik) azaltarak, hesaplamaları hafifletir.
    - Genellikle max pooling veya average pooling kullanılır.

## 4.7 Recurrent Layers

- **Açıklama:**
  - Zaman serileri, dil modelleme gibi sıralı verilerde kullanılan katmanlardır.
- **Örnekler:**
  - **LSTM (Long Short-Term Memory):**
    - Uzun süreli bağımlılıkları öğrenmede kullanılan gelişmiş bir RNN türüdür.
  - **GRU (Gated Recurrent Unit):**
    - LSTM’ye alternatif olarak, daha az parametre ile benzer performansı sağlar.

## 4.8 Normalization Layers

- **Açıklama:**
  - Eğitim sırasında katman çıktılarının normalize edilmesini sağlar.
  - Bu, gradyanların dengelenmesine ve öğrenme sürecinin stabil kalmasına yardımcı olur.
- **Örnek:**
  - **BatchNormalization:** Her mini-batch sonunda veriyi normalize ederek, öğrenmeyi hızlandırır.

## 4.9 Regularization Layers

- **Açıklama:**
  - Modelin aşırı öğrenmesini önlemek için kullanılan ek katmanlardır.
- **Örnek:**
  - **Dropout:** Rastgele bazı nöronları devre dışı bırakarak, modelin genelleme yeteneğini artırır.

## 4.10 Attention Layers

- **Açıklama:**
  - Modelin girdinin belirli kısımlarına daha fazla odaklanmasını sağlar.
  - Özellikle doğal dil işleme ve görüntü işleme gibi alanlarda kullanılır.
- **Detaylar:**
  - Bu mekanizma, modelin hangi özelliklere daha çok dikkat etmesi gerektiğini belirler.
  - Böylece, önemli bilgilerin daha net işlenmesi sağlanır.

## 4.11 Reshaping Layers

- **Açıklama:**
  - Girdinin boyutlarının değiştirilmesi işlemini gerçekleştirir.
  - Örneğin, çok boyutlu veriyi düzleştirerek veya yeniden şekillendirerek modelin girişine uygun hale getirir.

## 4.12 Merging Layers

- **Açıklama:**
  - Farklı katmanlardan gelen verileri birleştirmeyi sağlar.
  - Bu yapı, çoklu giriş veya dallanmış ağ yapılarında kullanılır.

####################################################################################################
# 5. Callbacks API

Callbacks, model eğitimi sırasında belirli olaylar gerçekleştiğinde otomatik olarak tetiklenen fonksiyonlardır.
Eğitim sürecinde çeşitli aksiyonları (örneğin, kontrol noktası oluşturma, erken durdurma) otomatik olarak gerçekleştirir.

## 5.1 Base Callback Class

- **Açıklama:**
  - Tüm callback’lerin temelini oluşturan sınıftır.
  - Kullanıcı kendi callback fonksiyonlarını bu sınıftan türeterek oluşturabilir.

## 5.2 Model Checkpoint

- **Açıklama:**
  - Eğitim süreci boyunca modelin ağırlıklarını düzenli aralıklarla veya performans arttığında kaydeder.
  - Ani bir kesinti veya hata durumunda, kaydedilen modelden devam edebilmenizi sağlar.

## 5.3 BackupAndRestore

- **Açıklama:**
  - Eğitim sürecinin yedeğini alır ve olası kesinti durumlarında yeniden yükleme yaparak eğitimin devamını sağlar.

## 5.4 EarlyStopping

- **Açıklama:**
  - Modelin belirli bir süre boyunca iyileşme göstermemesi durumunda eğitimi otomatik olarak durdurur.
  - Bu sayede, gereksiz yere eğitim süresinin uzaması önlenir.

## 5.5 LearningRateScheduler

- **Açıklama:**
  - Eğitim ilerledikçe öğrenme oranını (learning rate) önceden belirlenmiş bir plana göre ayarlar.
  - Bu, modelin farklı eğitim aşamalarında farklı öğrenme hızlarına adapte olmasını sağlar.

## 5.6 ReduceLROnPlateau

- **Açıklama:**
  - Modelin performansında durağanlık gözlemlendiğinde otomatik olarak öğrenme oranını düşürür.
  - Böylece, modelin ince ayarlarla daha iyi öğrenmesi hedeflenir.

## 5.7 RemoteMonitor

- **Açıklama:**
  - Eğitim sürecini uzaktan (örneğin, bulut tabanlı sistemlerden) izlemeye olanak tanır.
  - Gerçek zamanlı bildirimler ve takip için kullanılır.

## 5.8 LambdaCallback

- **Açıklama:**
  - Kısa ve öz callback fonksiyonlarını lambda ifadeleri ile tanımlamanızı sağlar.
  - Bu, basit olaylar için ekstra sınıf tanımlama gerektirmeden işlevsellik eklemenize imkan tanır.

## 5.9 TerminateOnNaN

- **Açıklama:**
  - Eğitim sırasında `NaN` (Not a Number) değer oluştuğunda eğitimi hemen sonlandırır.
  - Bu, hatalı hesaplamaların modelin genel eğitimine zarar vermesini engeller.

## 5.10 CSVLogger

- **Açıklama:**
  - Eğitim sürecinde elde edilen metrikleri ve diğer bilgileri CSV dosyasına kaydeder.
  - Böylece, eğitim sonrası verilerin analiz edilmesi kolaylaşır.

## 5.11 ProgbarLogger

- **Açıklama:**
  - Eğitim sırasında ilerlemeyi ve performans bilgilerini görsel olarak bir ilerleme çubuğu (progress bar) üzerinden sunar.
  - Bu, eğitim sürecini adım adım takip etmeyi kolaylaştırır.

## 5.12 SwapEMAWeights

- **Açıklama:**
  - Model ağırlıklarının üssel hareketli ortalamasını (exponential moving average) kullanarak, eğitim sonrası daha stabil sonuçlar elde etmeyi amaçlar.
  - Detayları henüz tam olarak netleşmemiş olsa da, bazı uygulamalarda model performansını artırmak için kullanılmaktadır.

####################################################################################################
# 6. Ops API

Ops API, Keras’ın temel matematiksel işlemleri ve lineer cebir fonksiyonlarını içerir.
Bu API sayesinde, NumPy tabanlı işlemler TensorFlow gibi backend’ler üzerinde etkili bir şekilde çalıştırılır.

**Özellikler:**
- Matris ve vektör işlemleri
- Lineer cebir hesaplamaları
- Diğer matematiksel fonksiyonlar

Bu sayede, karmaşık hesaplamalar daha verimli bir şekilde yapılabilir.

####################################################################################################
# 7. Optimizers (Optimizasyon Yöntemleri)

Optimizers, modelin ağırlıklarını güncelleyen algoritmalardır.
Her optimizasyon yöntemi, farklı hesaplama stratejileri kullanarak modelin öğrenme sürecini etkiler.

Aşağıda, en sık kullanılan optimizasyon algoritmaları detaylı olarak açıklanmıştır:

## 7.1 SGD (Stochastic Gradient Descent)

- **Açıklama:**
  - Model parametrelerini, her seferinde rastgele seçilen bir mini-batch üzerinden günceller.
  - Temel ve klasik bir optimizasyon algoritmasıdır.
- **Detaylar:**
  - Öğrenme oranı (learning rate) sabit veya değişken olabilir.
  - Büyük veri setlerinde hesaplama verimliliği sağlar.

## 7.2 RMSprop

- **Açıklama:**
  - Geçmiş gradyanların ortalamasını dikkate alarak her parametre için ayrı öğrenme oranı ayarlar.
- **Detaylar:**
  - Özellikle RNN gibi tekrarlayan yapılar için uygundur.
  - Gradyanların aşırı salınımını azaltır.

## 7.3 Adam (Adaptive Moment Estimation)

- **Açıklama:**
  - Hem gradyanların ortalamasını hem de karelerinin ortalamasını kullanarak adaptif öğrenme oranı belirler.
- **Detaylar:**
  - Genellikle hızlı yakınsama ve stabil eğitim sağlar.
  - Hem düşük hem yüksek öğrenme oranlarına karşı dayanıklıdır.

## 7.4 AdamW

- **Açıklama:**
  - Adam algoritmasının bir varyantıdır; ağırlık düşürme (weight decay) cezası ekler.
- **Detaylar:**
  - Modelin genelleme yeteneğini artırır.
  - Aşırı büyük ağırlık değerlerinin önüne geçer.

## 7.5 Adadelta

- **Açıklama:**
  - Öğrenme oranını otomatik olarak adapte eder, özellikle düşük başlangıç öğrenme oranlarında daha başarılıdır.
- **Detaylar:**
  - Ağırlık güncellemelerinde geçmiş bilgilere dayalı dinamik ayar yapar.

## 7.6 Adagrad

- **Açıklama:**
  - Öğrenme hızını her parametre için ayrı ayarlayarak, seyrek (sparse) verilerde avantaj sağlar.
- **Detaylar:**
  - Daha az güncellenen parametrelere yüksek öğrenme oranı verir.

## 7.7 Adamax

- **Açıklama:**
  - Adam algoritmasının sonsuz norm (infinity norm) kullanarak yapılan versiyonudur.
- **Detaylar:**
  - Bazı durumlarda daha kararlı sonuçlar verir.

## 7.8 Adafactor

- **Açıklama:**
  - Bellek kullanımını optimize eden bir optimizasyon algoritmasıdır.
- **Detaylar:**
  - Büyük modellerde hafıza tasarrufu sağlar.

## 7.9 Nadam

- **Açıklama:**
  - Nesterov momentum ile Adam algoritmasının birleşimidir.
- **Detaylar:**
  - İki algoritmanın avantajlarını birleştirir, daha hızlı yakınsama sağlayabilir.

## 7.10 FTRL (Follow The Regularized Leader)

- **Açıklama:**
  - Özellikle çevrimiçi (online) öğrenme uygulamalarında tercih edilen, düzenlileştirici özellikli bir optimizasyondur.
- **Detaylar:**
  - Sürekli veri akışında model güncellemeleri için kullanışlıdır.

## 7.11 Lion

- **Açıklama:**
  - Modelin öğrenme sürecini hızlandıran ve daha stabil hale getiren nispeten yeni bir optimizasyon yöntemidir.
- **Detaylar:**
  - Deneysel sonuçlarda hız ve kararlılık avantajları sunabilir.

## 7.12 Nesterov

- **Açıklama:**
  - Standart momentum yöntemine ek olarak “ön bakış” (lookahead) yaparak daha verimli güncelleme sağlar.
- **Detaylar:**
  - Modelin gelecekteki eğilimlerini tahmin eder ve buna göre adım atar.

## 7.13 Loss Scale Opt

- **Açıklama:**
  - Karışık hassasiyet (mixed precision) eğitiminde, sayısal kararlılığı sağlamak için kullanılan bir tekniktir.
- **Detaylar:**
  - Özellikle düşük değerli (float16 gibi) veri tiplerinde kullanılarak eğitim stabilitesini artırır.

####################################################################################################
# 8. Metrics (Başarı Ölçümleri)

Metrikler, modelin eğitim ve test performansını ölçmek için kullanılır.
Doğru metriklerin seçimi, modelin hangi alanlarda başarılı ya da başarısız olduğunu anlamak için kritik öneme sahiptir.

## 8.1 Base Metric Class

- **Açıklama:**
  - Kendi metriklerinizi tanımlamak için kullanılan temel sınıftır.

## 8.2 Accuracy (Doğruluk) Metric

- **Açıklama:**
  - Modelin doğru tahmin ettiği örneklerin oranını hesaplar.
- **Detaylar:**
  - Sınıflandırma problemlerinde en yaygın kullanılan metriklerden biridir.

## 8.3 Probabilistic Metrics

- **Açıklama:**
  - Modelin çıktısındaki olasılık dağılımlarını ve belirsizlikleri ölçer.
- **Detaylar:**
  - Çapraz entropi (cross-entropy) gibi metrikler bu kategoriye girer.

## 8.4 Regression Metrics

- **Açıklama:**
  - Regresyon problemlerinde, tahminler ile gerçek değerler arasındaki farkı ölçer.
- **Örnekler:**
  - **MSE (Mean Squared Error):** Hata karelerinin ortalaması.
  - **RMSE (Root Mean Squared Error):** MSE’nin karekökü.

## 8.5 Classification Metrics

- **Açıklama:**
  - Sınıflandırma problemlerinde modelin performansını ölçen metriklerdir.
- **Örnekler:**
  - **AUC (Area Under Curve):** ROC eğrisi altındaki alan.
  - **Recall (Duyarlılık):** Gerçek pozitiflerin oranı.
  - **Precision (Kesinlik):** Pozitif tahminlerin doğruluk oranı.
  - **F1 Score:** Precision ve Recall’un harmonik ortalaması.

## 8.6 Image Segmentation Metrics

- **Açıklama:**
  - Görüntü segmentasyonu problemlerinde, modelin bölge tahminlerinin doğruluğunu ölçer.
- **Örnek:**
  - **IoU (Intersection over Union):** Tahmin edilen ve gerçek segmentasyon alanlarının kesişim/birleşim oranı.

## 8.7 Metric Wrappers and Reduction Metrics

- **Açıklama:**
  - Birden fazla metriğin birleşimi veya indirgenmesi için kullanılan yardımcı araçlardır.
  - Karmaşık metrik hesaplamalarında, tekil bir metrik elde etmek için kullanılır.

####################################################################################################
# 9. Available Losses (Kayıp Fonksiyonları)

Loss fonksiyonları, modelin tahminleri ile gerçek değerler arasındaki farkı ölçerek modelin “yanlış” yaptığı miktarı belirler.

## 9.1 Probabilistic Losses

- **Açıklama:**
  - Olasılık dağılımlarını kullanarak hata hesaplaması yapar.
- **Örnek:**
  - Çapraz entropi (cross-entropy) loss, sınıflandırma problemlerinde yaygın olarak kullanılır.

## 9.2 Regression Losses

- **Açıklama:**
  - Regresyon problemlerinde kullanılan, tahmin hatalarını ölçen loss fonksiyonlarıdır.
- **Örnek:**
  - **MSE (Mean Squared Error):** Hata karelerinin ortalaması.

## 9.3 Hinge Loss

- **Açıklama:**
  - Maksimum margin (margin) sınıflandırmalarında kullanılan bir loss fonksiyonudur.
- **Detaylar:**
  - Destek Vektör Makineleri (SVM) gibi algoritmalarda tercih edilir.

####################################################################################################
# 10. Data Loading (Veri Yükleme)

Veri setlerinin, özellikle resim gibi büyük dosya gruplarının güvenli ve düzenli bir şekilde yüklenmesini sağlar.

**Özellikler:**
- Dosya sistemindeki klasör yapısının belirli bir düzene göre organize edilmesi.
- Otomatik etiketleme ve veri ön işleme araçları.
- Büyük veri setlerinde, veri akışını optimize etmek için asenkron yükleme desteği.

Bu özellikler, veri setlerinin hatasız ve verimli bir şekilde modele sunulmasını sağlar.

####################################################################################################
# 11. Mixed Precision (Karışık Hassasiyet)

Mixed precision, model eğitiminde kullanılan veri tiplerini optimize ederek hem bellek tasarrufu hem de hız artışı sağlar.

**Nasıl Çalışır?**
- Örneğin, 32-bit floating point (float32) yerine 16-bit floating point (float16) kullanılarak hesaplama yoğunluğu azaltılır.
- GPU’ların bu tip hesaplamaları daha hızlı yapabilme özelliğinden yararlanılır.

**Faydaları:**
- Eğitim süresinde hızlanma.
- Bellek kullanımında ciddi oranda tasarruf.
- Büyük modellerin daha düşük donanım gereksinimleriyle eğitilebilmesi.

####################################################################################################
# 12. RNG API (Random Number Generator - Rastgele Sayı Üreteci)

RNG API, modelinizdeki rastgelelik işlemlerini yönetir.

**Özellikler:**
- Rastgele sayı üretimi: Modelin başlatılması, veri karıştırma (shuffling) ve diğer rastgele işlemler.
- Yeniden üretilebilirlik: Aynı random seed kullanılarak, deneylerin tekrar edilebilir olması sağlanır.

Bu, deneylerin güvenilirliğini ve tekrarlanabilirliğini artırır.

####################################################################################################
# 13. Utilities (Yardımcı Araçlar)

Keras, model ve veri işleme süreçlerini kolaylaştıran birçok yardımcı araç sunar.

## 13.1 Model Plotting Utilities

- **Açıklama:**
  - Modelin yapısını grafiksel olarak görselleştirir.
  - Bu araç sayesinde, modelin katmanları, bağlantıları ve parametreleri anlaşılır bir şekilde sunulur.

## 13.2 Structured Data Preprocessing Utilities

- **Açıklama:**
  - Tablo veya yapılandırılmış verilerin ön işleme adımlarını otomatikleştirir.
  - Özellikle veri temizleme, normalizasyon ve özellik mühendisliği (feature engineering) süreçlerinde kullanılır.

## 13.3 Tensor Utilities

- **Açıklama:**
  - Çok boyutlu dizi (tensor) işlemlerinde yardımcı fonksiyonlar sunar.
  - Tensor dönüşümleri, şekil değiştirme (reshaping) ve hesaplama işlemleri burada ele alınır.

####################################################################################################
# 14. KerasTuner

KerasTuner, modelinizin hiperparametrelerini otomatik olarak aramak ve optimize etmek için kullanılan gelişmiş bir araçtır.

**Ana Bileşenleri:**

## 14.1 Hyperparameter

- **Açıklama:**
  - Modelin öğrenme sürecinde ayarlanabilir parametrelerdir.
  - Öğrenme oranı, dropout oranı, katman sayısı gibi değerler hiperparametre örnekleridir.

## 14.2 Tuners

- **Açıklama:**
  - Farklı hiperparametre kombinasyonlarını deneyerek en iyi sonucu bulmaya çalışırlar.
  - Arama stratejileri (örneğin, rastgele arama, bayesian optimizasyon) uygulanır.

## 14.3 Oracles

- **Açıklama:**
  - Denenecek hiperparametre kombinasyonlarını belirler.
  - Hangi kombinasyonların daha verimli olacağını “tahmin eder” ve aramayı yönlendirir.

## 14.4 Hypermodels

- **Açıklama:**
  - Hiperparametre aramasına uygun model yapılarını otomatik olarak oluşturur.
  - Farklı model konfigürasyonlarını deneyerek en iyi mimariyi bulmayı amaçlar.

## 14.5 Errors

- **Açıklama:**
  - Hiperparametre optimizasyonu sırasında oluşabilecek hataların yönetilmesi ve raporlanması için kullanılan modüldür.

## 14.6 KerasCV ve KerasNLP

- **Açıklama:**
  - Sırasıyla bilgisayarlı görü (image processing) ve doğal dil işleme (NLP) alanları için optimize edilmiş Keras API'leridir.
  - Bu alanlardaki modellerin, veri işleme ve eğitim süreçlerinin optimize edilmesini sağlar.

####################################################################################################
# 15. Popüler Modeller ve Mimari Örnekleri

Keras ile geliştirilen ve yaygın olarak kullanılan bazı popüler derin öğrenme modelleri aşağıda detaylı olarak açıklanmıştır.

## 15.1 Xception

- **Açıklama:**
  - Derin ayrık evrişim (depthwise separable convolution) kullanarak modelin parametre sayısını azaltırken performansı artırır.
- **Detaylar:**
  - Görüntü işleme görevlerinde yüksek verimlilik sunar.
  - Standart konvolüsyonel katmanlara göre daha az hesaplama gerektirir.

## 15.2 EfficientNet

- **Açıklama:**
  - Modelin genişliği (width), derinliği (depth) ve çözünürlüğünü (resolution) aynı anda optimize eden dengeli bir mimaridir.
- **Detaylar:**
  - Farklı boyutlarda versiyonları bulunur (EfficientNet-B0’den B7’ye kadar).
  - Performans ve verimlilik arasında optimal denge sağlar.

## 15.3 ConvNeXt

- **Açıklama:**
  - Klasik konvolüsyonel mimarinin modern tekniklerle iyileştirilmiş halidir.
- **Detaylar:**
  - Görüntü işleme görevlerinde yüksek performans sunar.
  - Yeni optimizasyon teknikleri ile daha kararlı eğitim sağlar.

## 15.4 VGG16 & VGG19

- **Açıklama:**
  - 16 veya 19 katmandan oluşan derin konvolüsyonel sinir ağlarıdır.
- **Detaylar:**
  - Görüntü sınıflandırma problemlerinde yaygın olarak kullanılır.
  - Basit ve anlaşılır yapıdadır; ancak parametre sayısı oldukça yüksektir.

## 15.5 ResNet

- **Açıklama:**
  - "Residual Network" olarak adlandırılır; katmanlar arasında atlama (skip connection) bağlantıları bulunur.
- **Detaylar:**
  - Çok derin ağlarda gradyan kaybolmasını önler.
  - Eğitim sürecini hızlandırır ve doğruluğu artırır.

## 15.6 MobileNet

- **Açıklama:**
  - Mobil ve gömülü sistemlerde kullanılmak üzere optimize edilmiş hafif bir mimaridir.
- **Detaylar:**
  - Düşük hesaplama gücü ve bellek kullanımı ile yüksek performans sunar.
  - Gerçek zamanlı uygulamalarda tercih edilir.

## 15.7 DenseNet

- **Açıklama:**
  - Her katmanın, kendisinden önceki tüm katmanlara doğrudan bağlı olduğu bir yapıdır.
- **Detaylar:**
  - Bilgi akışını ve gradient propagasyonunu iyileştirir.
  - Daha az parametre ile yüksek performans elde edilir.

## 15.8 NasNet

- **Açıklama:**
  - Neural Architecture Search (NAS) kullanılarak otomatik olarak tasarlanan bir mimaridir.
- **Detaylar:**
  - Hesaplama kaynaklarına göre en optimal yapıyı oluşturur.
  - Farklı versiyonları, çeşitli cihaz ve uygulamalara uyacak şekilde optimize edilmiştir.

## 15.9 Inception

- **Açıklama:**
  - Google tarafından geliştirilen ve faktörize edilmiş (factorized) evrişim teknikleri kullanan bir modeldir.
- **Detaylar:**
  - Hesaplama maliyetini düşürürken, modelin derinliğini ve genişliğini optimize eder.
  - Inception mimarisinin, ResNet ile birleştirilmiş versiyonları da mevcuttur.

####################################################################################################
# Sonuç ve Ek Bilgiler

Bu dosya, Keras API’lerinin tüm temel ve ileri düzey bileşenlerini yeni başlayanların anlayabileceği şekilde açıklamaya yönelik olarak hazırlanmıştır.
Her bir bölüm, ilgili konuyu derinlemesine ele alır ve örneklerle desteklenir.

Ekstra öneriler:
- Her bölümde yer alan kavramları kendi projelerinizde deneyerek pekiştirmeye çalışın.
- Resmi Keras dokümantasyonunu ve ilgili topluluk forumlarını takip ederek, güncel gelişmelerden haberdar olun.
- Kod örneklerini çalıştırarak, her bir metot ve fonksiyonun işleyişini adım adım gözlemleyin.

Unutmayın:
- Derin öğrenme ve sinir ağları konusu oldukça geniştir.
- Temel kavramları iyi anladığınızda, daha karmaşık modelleri ve mimarileri oluşturmanız çok daha kolay olacaktır.

Bu rehberde sunulan bilgiler, her ne kadar kapsamlı ve detaylı olsa da, Keras ekosistemindeki gelişmeler doğrultusunda zaman içinde güncellenmeye ihtiyaç duyabilir.
Dolayısıyla, yeni çıkan kütüphane versiyonları ve güncellemeler ile birlikte öğrenmeye devam etmek önemlidir.

# Son Söz

Keras API'leri ile ilgili bu rehber, derin öğrenme alanında yeni başlayanlar için temel ve ileri düzey bilgileri bir araya getirmektedir.
Her bölüm, ilgili konuyu detaylandırarak, örnekler ve açıklamalar ile desteklenmiştir.

Bu rehberi okuduktan sonra:
- Model tanımlama ve eğitme süreçlerinde hangi adımları takip edeceğinizi daha iyi anlayacaksınız.
- Farklı katman tiplerinin, callback mekanizmalarının ve optimizasyon yöntemlerinin işlevlerini kavrayabileceksiniz.
- Uygulamalı örnekler üzerinden, Keras ile kendi projelerinizi geliştirmeye başlayabileceksiniz.

Öğrenme sürecinizde başarılar dileriz; unutmayın ki, her yeni bilgi, sizi daha ileriye taşıyan bir adımdır.

####################################################################################################
