---
layout: post
title: "Makine Öğrenmesi İçin Teorik Rehber"
subtitle: "Yeni Başlayanlar İçin Makine Öğrenmesi"
cover-img: /assets/img/yapay-zeka-cover.webp
thumbnail-img: /assets/img/yapay-zeka-share.webp
share-img: /assets/img/yapay-zeka-share.webp
tags: [makine öğrenmesi, makine, yazılım, python, ai]
author: Tunahan Yardımcı
date: 2025-02-24 12:01
---

# Makine Öğrenmesi Rehberi

Bu doküman, makine öğrenmesinin temel kavramlarını, algoritmalarını ve yöntemlerini
kapsamlı bir biçimde ele almaktadır. Aşağıda, çeşitli makine öğrenmesi konularını detaylı
biçimde inceleyebilirsiniz.

---

## 1. Öğrenme Yöntemleri

Makine öğrenmesi farklı yaklaşımlarla uygulanabilir:

### 1.1 Denetimli Öğrenme
- **Tanım:** Giriş ve çıkış verilerinin etiketli örnekler üzerinden öğrenilmesi.
- **Örnekler:**
  - Ortalama ev fiyatlarının m² başına hesaplanması.
  - Tümör boylarının istatistiksel analizi ve sınıflandırılması.

### 1.2 Denetimsiz Öğrenme
- **Tanım:** Sadece giriş verileri kullanılarak, veriler arasındaki gizli desenlerin keşfi.
- **Örnek:**
  - Google Haberler’in, kelime bağlarını kullanarak haberleri gruplayıp sunması.

### 1.3 Yarı Denetimli Öğrenme
- **Tanım:** Az etiketli ve çok etiketsiz verinin kombinasyonu ile öğrenmenin sağlanması.
- **Örnek:**
  - Google Photos’un, fotoğrafları önce gruplayıp ardından az sayıda etiketle sınıflandırması.

### 1.4 Kendi Kendini Denetleyen Öğrenme
- **Tanım:** Modelin, otomatik etiketleme yaparak eksik bilgileri tamamlaması.

### 1.5 Transfer Learning
- **Tanım:** Bir görevde elde edilen bilginin, farklı bir göreve aktarılması.

### 1.6 Reinforcement Learning (Pekiştirmeli Öğrenme)
- **Tanım:** Ajanın, çevresiyle etkileşime girerek en iyi ödülü alacak stratejiyi öğrenmesi.
- **Örnek:**
  - Robotların yürüme gibi davranışları deneyerek öğrenmesi.

---

## 2. Doğrusal Regresyon Modeli ve Yaklaşımlar

**Doğrusal Regresyon** iki yaklaşıma ayrılır:

- **Instance-based Learning:** 
  - Verilerin örnek bazında ezberlenerek öğrenilmesi; örneğin spam filtrelemede yetersiz kalabilir.

- **Model-based Learning:** 
  - Verilerden model oluşturularak tahmin yapılması.

> **Dipnot:**  
> Model terimi üç şekilde yorumlanabilir:
> - **Model Türü:** Örneğin, doğrusal regresyon veya karar ağacı.
> - **Model Mimarisi:** Giriş ve çıkış detayları.
> - **Eğitilmiş Model:** Tahmine hazır, eğitim süreci tamamlanmış model.

Model parametreleri:
- **w:** Ağırlıklar
- **b:** Bias (sapma)

---

## 3. Eğitimde Gerekli Temel Kavramlar

Temel kavramlar:
- **Label:** Veri etiketleri (örn. ticket bilgisi)
- **Feature:** Girdi özellikleri
- **Model:** $ x + f(x) + y $
- **Doğrulama:** Model performansının ölçülmesi
- **Sınıflandırma:** Doğru etiket tahmini
- **Regresyon:** Sayısal çıktı tahmini (örn. ev fiyat tahmini)

Ek kavramlar:
- Öznitelik mühendisliği, hiperparametreler, overfitting, underfitting,
  cross-validation, derin öğrenme.

Kullanılan kütüphaneler:
- **matplotlib, pandas, scikitlearn, numpy, tensorflow, pytorch, keras, theano**

---

## 4. Veri Ön İşleme

Veri ön işleme adımları:

- **Veri Temizleme:** Eksik, hatalı ve tutarsız verilerin düzeltilmesi.
- **Standardizasyon & Normalizasyon:** Farklı ölçeklerdeki verilerin ortak forma getirilmesi.
- **Öznitelik Seçimi:** En bilgilendirici özniteliklerin belirlenmesi.
- **Veri Kodlaması:** Kategorik verilerin sayısal formata dönüştürülmesi.
- **Öznitelik Çıkarma:** Yeni verilerin türetilmesi (örn. doğum tarihinden yaş çıkarma).

Ek olarak, istatistiksel ölçümler (minimum, maximum, ortalama, sapma) hesaplanır.

Veri setleri genellikle eğitim, doğrulama ve test olarak bölünür.

---

## 5. Gözetimli Öğrenme Yöntemleri

**Algoritmalar:**

- **K-En Yakın Komşu (KNN):**
  - **Mesafe Ölçümleri:** Euclidean, Manhattan, Minkowski, Hamming.
  - **Avantaj:** Basit ve hızlı.
  - **Dezavantaj:** Veri boyutu ve özellik duyarlılığı.

- **Karar Ağaçları:**
  - Türleri: ID3, C4.5, CART (Gini karışıklığı kullanılır).

- **Ensemble Learning:**
  - Rastgele Orman: Birden fazla karar ağacının toplu kullanımı.

- **Lojistik Regresyon:**
  - İleri yayılım, loss hesaplama ve geri yayılım (gradient descent).

- **Destek Vektör Makineleri (SVM):**
  - Kernel trick yöntemiyle veriyi doğrusal ayrılabilir hale getirme.

- **Naive Bayes:**
  - Bayes teoremine dayalı sınıflandırma.

---

## 6. Lineer Regresyon

**Model Türleri:**

- **Basit Lineer Regresyon:** $ y = ax + b $
- **Çoklu Lineer Regresyon:** $ y = b_0 + b_1x_1 + b_2x_2 + \dots $
- **Polinom Regresyon:** $ y = b_0 + b_1x + b_2x^2 + \dots $

**Örnek Hesaplama:**

- $ x $ değerleri: 1, 2, 3, 4, 5
- $ y $ değerleri: 1.2, 1.8, 2.6, 3.2, 3.8
- Hesaplanan katsayılar:
  - $ a \approx 0.66 $
  - $ b \approx 0.54 $
- Model: $ y = 0.66x + 0.54 $

Avantaj: Basit, eğitim süresi kısa, yorumlanabilir.
Dezavantaj: Doğrusal olmayan ilişkilerde yetersiz kalır.

---

## 7. Gözetimsiz Öğrenme

**Yaklaşımlar:**

- **K-Means Kümeleme**
- **Hiyerarşik Kümeleme:**
  - Türler: Aglomeratif ve Bölücü.
- **DBSCAN:**
  - Yoğunluk temelli kümeleme; Hiperparametreler: MinPoint, Epsilon.

---

## 8. Pekiştirmeli Öğrenme

**Temel Kavramlar:**

- Agent, Environment, Action, State, Reward, Policy, Value, Q-Value.

**Yöntemler:**

- Model tabanlı veya model tabanlı olmayan,
- Değer tabanlı (Q-Learning) ve Deep Q-Learning.

**Güncelleme Formülü:**

$ Q(s,a) = Q(s,a) + \alpha \, [R + \gamma \max_{a'} Q(s',a') - Q(s,a)] $

---

## 9. Boyut İndirgeme

**Teknikler:**

- Öznitelik seçimi ve öznitelik çıkarma.
- **PCA (Temel Bileşen Analizi):**
  - Veri normalizasyonu, kovaryans matrisinin hesaplanması,
    özdeğer ve özvektör analizi, yeni veri setinin oluşturulması.
- **LDA (Doğrusal Diskriminant Analizi):**
  - Sınıflar arasındaki ayrımın maksimize edilmesi.
- **t-SNE:**
  - Yüksek boyutlu veriyi görselleştirmek için düşük boyuta indirgeme.

---

## 10. Değerlendirme Metrikleri

### Sınıflandırma

- Doğruluk, Kesinlik (Precision), Duyarlılık (Recall),
  F1-Score, ROC Eğrisi & AUC, Karmaşıklık Matrisi.

### Regresyon

- Ortalama Mutlak Hata (MAE), Ortalama Kare Hata (MSE),
  Kök Ortalama Kare Hata (RMSE), R-Kare (R²).

---

## 11. Model Seçimi ve Doğrulama

Model seçimi; problem türü, veri kalitesi, hesaplama kaynakları,
yorumlanabilirlik gibi faktörlere bağlıdır.

**Yöntemler:**

- **Probabilistic Yöntemler:**
  - Akaike Bilgi Kriteri (AIC),
  - Minimum Açıklama Uzunluğu,
  - Bayesian Bilgi Kriteri (BIC).

- **Rastgele Bölünme Yöntemleri:**
  - Bootstrap,
  - K-Fold Çapraz Doğrulama,
  - Leave-One-Out (LOO).

---

## 12. Regülarizasyon ve Hiperparametre Ayarı

**Regülarizasyon Teknikleri:**

- **Lasso (L1):** Bazı katsayıları sıfıra indirger.
- **Ridge (L2):** Katsayıları küçültür fakat sıfıra çekmez.
- **ElasticNet:** L1 ve L2’nin birleşimi.

**Hiperparametre Ayarlaması:**

- Literatür taraması, manuel arama, grid search ve random search yöntemleri.

-------------------------------------------------------------------
