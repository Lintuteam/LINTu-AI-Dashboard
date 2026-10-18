# 🚀 LİNTu AI Dashboard

> **Görüntü İşleme, LLM (RAG) ve Veri Görselleştirme Destekli Bütünleşik Yapay Zeka Yönetim Paneli**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-000000?style=for-the-badge&logo=ultralytics&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🌐 Canlı Yayın (Live Demo)

Uygulamayı herhangi bir kurulum yapmadan doğrudan tarayıcınızda deneyimleyebilirsiniz:  
👉 **[LİNTu AI Dashboard Canlı Uygulama](https://deploy-rquvqmdqh3m6kryrnzztf5.streamlit.app)**

---

## 📌 Proje Hakkında

**LİNTu AI Dashboard**, LİNTu Team tarafından geliştirilen ve saha verilerini anlık olarak analiz edip karara dönüştüren çok modüllü bir yapay zeka platformudur. Platform; görüntü işleme modelleri (YOLOv8/OpenCV), büyük dil modelleri (Gemini API) ve etkileşimli harita/grafik araçlarını tek bir arayüzde birleştirir.

---

## 🛠️ Ana Modüller ve Özellikler

### 👁️ 1. Görsel Ön İşleme & Yapay Zeka Analiz
- **YOLOv8 Nesne Tespiti:** Yüklenen görseller veya kamera akışı üzerinde gerçek zamanlı nesne tespiti.
- **OpenCV Filtreleme:** Görüntü netleştirme, eşikleme (thresholding) ve ön işleme adımları.
- **Gemini LLM Otomatik Raporlama:** Tespit edilen nesne ve risk durumlarını otomatik olarak anlamlandırıp Türkçe/İngilizce özet rapor üretme.

### 📊 2. Dinamik Grafik & Harita Paneli
- **Plotly Görselleştirme:** Veri dağılımlarını anlık gösteren etkileşimli çubuk ve pasta grafikleri.
- **Folium Harita Entegrasyonu:** Konum bazlı veri analizleri ve coğrafi ısı haritaları.

### 📁 3. CSV Veri İşleme & İndirme
- **Otomatik Veri Temizleme:** Eksik ve hatalı kayıtların Pandas ile temizlenmesi.
- **Rapor Dışa Aktarımı:** İşlenen verilerin `.csv` veya Excel formatında indirilebilir hale getirilmesi.

---

## 💻 Teknoloji Yığını (Tech Stack)

| Alan | Kullanılan Teknolojiler |
| :--- | :--- |
| **Ön Yüz & Deploy** | Streamlit, Streamlit Community Cloud |
| **Görüntü İşleme** | OpenCV, Ultralytics YOLOv8, Roboflow |
| **Yapay Zeka & LLM** | Google Gemini API (`google-genai`), RAG |
| **Veri & Harita** | Pandas, Plotly, Folium, `streamlit-folium` |
| **Sürüm Kontrolü** | Git, GitHub Organizations |

---

## 📂 Proje Klasör Yapısı

```text
LINTu-AI-Dashboard/
├── app.py              # Ana Streamlit uygulama çalıştırma dosyası
├── requirements.txt    # Sunucu bağımlılıkları ve kütüphaneler
├── .gitignore          # Repoya dahil edilmeyen geçici/gizli dosyalar
└── README.md           # Proje dokümantasyonu
