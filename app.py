import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

# ==========================================
# İSTASYON ENTEGRASYON FONKSİYONLARI
# Yusuf (İstasyon A) ve Tolga/Senih (İstasyon B) kodlarını buraya bağlayacak
# ==========================================
def run_yolo_detection(image):
    """İstasyon A (Yusuf & Tolga): YOLOv8 Görsel Analiz Modülü"""
    return {
        "object_count": 5,
        "risk_rate": 80,
        "confidence": 96.8
    }

def run_llm_analysis(image, detection_res):
    """İstasyon B (Tolga & Senih): Gemini / RAG Rapor Üretim Modülü"""
    return (
        f"Görsel üzerinde yapılan yapay zeka ön analizinde {detection_res['object_count']} adet "
        f"kritik unsur tespit edilmiştir. Sistem tarafından belirlenen ortalama Risk Oranı: %{detection_res['risk_rate']}."
    )

# ==========================================
# 1. SAYFA YAPILANDIRMASI
# ==========================================
st.set_page_config(
    page_title="LİNTu Team - Central AI Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# 2. SOL YAN MENÜ (SIDEBAR) NAVİGASYONU
# ==========================================
st.sidebar.title("⚡ LİNTu AI Dashboard")
st.sidebar.caption("İstasyon C: Tunahan & Alperen")

secim = st.sidebar.radio(
    "Modül Seçimi:",
    [
        "🖼️ Görsel Ön İşleme & AI Analiz",
        "📊 Dinamik Grafik & Harita Paneli",
        "📁 CSV Veri İşleme & İndirme",
        "👥 Takım Kadrosu & İstasyonlar"
    ]
)

st.sidebar.divider()
st.sidebar.success("🟢 Sistem Hazır")

# ==========================================
# MODÜL 1: GÖRSEL ÖN İŞLEME VE AI ANALİZ
# ==========================================
if secim == "🖼️ Görsel Ön İşleme & AI Analiz":
    st.title("🖼️ Görsel Ön İşleme & Yapay Zeka Analiz Paneli")
    st.caption("Görsel Filtreleme, YOLO Nesne Tanıma ve LLM Otomatik Raporlama")
    st.divider()

    uploaded_file = st.file_uploader("Analiz için bir görsel yükleyin (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        
        col_sol, col_sag = st.columns(2)

        # SOL KOLON: FİLTRELEME & GÖRSEL ÖN İŞLEME
        with col_sol:
            st.subheader("📸 Görsel Ön İşleme (Filtreler)")
            
            filtre = st.selectbox(
                "Model Öncesi Görsel Filtresi Seçin:",
                ["Orijinal", "Siyah-Beyaz (Grayscale)", "Kenar Tespiti (Edge Detection)", "Bulanıklaştırma (Blur)"]
            )
            
            if filtre == "Siyah-Beyaz (Grayscale)":
                processed_image = ImageOps.grayscale(raw_image)
            elif filtre == "Kenar Tespiti (Edge Detection)":
                processed_image = raw_image.filter(ImageFilter.FIND_EDGES)
            elif filtre == "Bulanıklaştırma (Blur)":
                processed_image = raw_image.filter(ImageFilter.BLUR)
            else:
                processed_image = raw_image

            st.image(processed_image, caption=f"Uygulanan Filtre: {filtre}", use_container_width=True)

        # SAĞ KOLON: MODEL ÇIKTILARI (FONKSİYONLARDAN ÇEKİLİR)
        with col_sag:
            st.subheader("🤖 Yapay Zeka Model Çıktıları")
            
            yolo_results = run_yolo_detection(processed_image)
            llm_report = run_llm_analysis(processed_image, yolo_results)

            st.markdown("##### 🎯 İstasyon A: Nesne Tanıma Metrikleri (YOLOv8)")
            m1, m2, m3 = st.columns(3)
            m1.metric(label="Tespit Edilen Nesne", value=f"{yolo_results['object_count']} Adet")
            m2.metric(label="Risk Oranı", value=f"%{yolo_results['risk_rate']}")
            m3.metric(label="Güven Skoru", value=f"%{yolo_results['confidence']}")

            st.divider()

            st.markdown("##### 📝 İstasyon B: Otomatik Rapor & Analiz (Gemini)")
            st.text_area(
                label="Model Analiz Raporu:",
                value=llm_report,
                height=120
            )
            
            if st.button("📩 Analiz Raporunu Kaydet"):
                st.toast("Analiz raporu başarıyla kaydedildi!")
    else:
        st.info("💡 Lütfen yukarıdaki alandan bir görsel yükleyin.")

# ==========================================
# MODÜL 2: DİNAMİK GRAFİK VE HARİTA PANELİ
# ==========================================
elif secim == "📊 Dinamik Grafik & Harita Paneli":
    st.title("📊 Dinamik Gösterge Paneli & Canlı Harita")
    st.caption("Canlı sistem metrikleri, Plotly grafikleri ve lokasyon haritası")
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Tespit Edilen Nesne", value="5 Adet", delta="+2")
    m2.metric(label="Risk Oranı", value="%80", delta="-5%")
    m3.metric(label="Model Doğruluğu", value="96.1%", delta="+1.2%")
    m4.metric(label="Yanıt Süresi", value="0.9 sn", delta="-0.3 sn")

    st.divider()

    st.subheader("📈 Performans & Veri Grafiği")
    secilen_gorunum = st.selectbox(
        "Görüntülemek istediğiniz veriyi seçin:",
        ["İstasyon İlerleme Oranları", "Haftalık Çalışma Saatleri", "Model Test Doğrulukları"]
    )

    if secilen_gorunum == "İstasyon İlerleme Oranları":
        chart_data = {"Kategori": ["İstasyon A (YOLO)", "İstasyon B (LLM)", "İstasyon C (Arayüz)", "İstasyon D (Veri)"], "Değer": [92, 85, 100, 80]}
    elif secilen_gorunum == "Haftalık Çalışma Saatleri":
        chart_data = {"Kategori": ["İstasyon A (YOLO)", "İstasyon B (LLM)", "İstasyon C (Arayüz)", "İstasyon D (Veri)"], "Değer": [55, 44, 40, 32]}
    else:
        chart_data = {"Kategori": ["YOLOv8 Nesne", "Gemini Prompt", "RAG Doküman", "Görsel İşleme"], "Değer": [96, 93, 91, 98]}

    df_dynamic = pd.DataFrame(chart_data)
    c1, c2 = st.columns(2)

    with c1:
        fig_bar = px.bar(df_dynamic, x="Kategori", y="Değer", color="Kategori", text_auto=True, template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        fig_pie = px.pie(df_dynamic, names="Kategori", values="Değer", hole=0.4, template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    st.subheader("🗺️ Canlı Lokasyon & Veri Haritası")
    map_data = pd.DataFrame({
        'lat': [41.0253, 41.0822, 41.0082],
        'lon': [28.8903, 29.0522, 28.9784]
    })
    st.map(map_data, zoom=10)

# ==========================================
# MODÜL 3: CSV VERİ İŞLEME VE İNDİRME
# ==========================================
elif secim == "📁 CSV Veri İşleme & İndirme":
    st.title("📁 CSV Veri Temizleme & İndirme Paneli")
    st.caption("Pandas ile CSV Okuma, Boş Satırları Temizleme (`df.dropna()`) ve Dışa Aktarma")
    st.divider()

    uploaded_csv = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])

    if uploaded_csv is not None:
        df_raw = pd.read_csv(uploaded_csv)
        
        st.subheader("📄 Ham Veri Tablosu (İlk 5 Satır)")
        st.dataframe(df_raw.head(), use_container_width=True)
        
        col_c1, col_c2 = st.columns(2)
        col_c1.info(f"**Toplam Satır Sayısı:** {len(df_raw)}")
        col_c2.info(f"**Eksik/Boş Veri Satırı Sayısı:** {df_raw.isnull().sum().sum()}")
        
        st.divider()
        
        if st.button("🧹 Boş Satırları Temizle (df.dropna())"):
            df_clean = df_raw.dropna()
            st.success(f"Temizleme tamamlandı! Yeni satır sayısı: {len(df_clean)}")
            st.dataframe(df_clean.head(), use_container_width=True)
            
            csv_data = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Temizlenmiş CSV Dosyasını İndir",
                data=csv_data,
                file_name="LINTu_Cleaned_Data.csv",
                mime="text/csv"
            )
    else:
        st.info("💡 İpucu: Kaggle'dan indirdiğiniz herhangi bir `.csv` dosyasını buraya sürükleyip bırakabilirsiniz.")

# ==========================================
# MODÜL 4: TAKIM KADROSU
# ==========================================
elif secim == "👥 Takım Kadrosu & İstasyonlar":
    st.title("⚡ LİNTu Yapay Zeka Takımı")
    st.caption("Rotasyonlu Çalışma Modeli ve Görev Dağılımı")
    st.divider()

    ekip_data = {
        "İsim": ["Yusuf", "Tolga", "Senih", "Tunahan", "Alperen", "Cemal & Tuna"],
        "Sorumlu Olduğu Alan": ["Görüntü İşleme", "LLM / Çeviri", "LLM / Çeviri", "Arayüz & Grafik", "Arayüz & Grafik", "Veri & Sunum"],
        "Kullandığı Araçlar": ["YOLOv8 / OpenCV", "Gemini API", "Prompt Eng.", "Streamlit / UI", "Plotly / Dashboard", "Kaggle / Pandas"]
    }
    st.table(pd.DataFrame(ekip_data))