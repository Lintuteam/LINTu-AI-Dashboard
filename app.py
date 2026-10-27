import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from src.vision_module import run_yolo_detection
from src.llm_module import run_llm_analysis

# ==========================================
# UI ARAYÜZ FONKSİYONLARI (İSTASYON C)
# ==========================================

def render_sidebar():
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
    st.sidebar.success("🟢 Entegrasyon Katmanı Aktif")
    return secim


def page_vision_analysis():
    st.title("🖼️ Görsel Ön İşleme & Yapay Zeka Analiz Paneli")
    st.caption("Görsel Filtreleme, YOLO Nesne Tanıma ve LLM Otomatik Raporlama")
    st.divider()

    uploaded_file = st.file_uploader("Analiz için bir görsel yükleyin (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        col_sol, col_sag = st.columns(2)

        with col_sol:
            st.subheader("📸 Görsel Ön İşleme (Filtreler)")
            filtre = st.selectbox("Model Öncesi Görsel Filtresi Seçin:", 
                                  ["Orijinal", "Siyah-Beyaz (Grayscale)", "Kenar Tespiti (Edge Detection)", "Bulanıklaştırma (Blur)"])
            
            if filtre == "Siyah-Beyaz (Grayscale)": processed_image = ImageOps.grayscale(raw_image)
            elif filtre == "Kenar Tespiti (Edge Detection)": processed_image = raw_image.filter(ImageFilter.FIND_EDGES)
            elif filtre == "Bulanıklaştırma (Blur)": processed_image = raw_image.filter(ImageFilter.BLUR)
            else: processed_image = raw_image

            st.image(processed_image, caption=f"Uygulanan Filtre: {filtre}", use_container_width=True)

        with col_sag:
            st.subheader("🤖 Yapay Zeka Model Çıktıları")
            
            # Arka plan entegrasyon fonksiyonlarını çağırıyoruz
            yolo_results = run_yolo_detection(processed_image)
            llm_report = run_llm_analysis(processed_image, yolo_results)

            st.markdown("##### 🎯 İstasyon A: Nesne Tanıma Metrikleri (YOLOv8)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Tespit Edilen Nesne", f"{yolo_results['object_count']} Adet")
            m2.metric("Risk Oranı", f"%{yolo_results['risk_rate']}")
            m3.metric("Güven Skoru", f"%{yolo_results['confidence']}")
            st.divider()

            st.markdown("##### 📝 İstasyon B: Otomatik Rapor (Gemini)")
            st.text_area("Model Analiz Raporu:", value=llm_report, height=140)
            
            if st.button("📩 Analiz Raporunu Kaydet"):
                st.toast("Analiz raporu başarıyla kaydedildi!")
    else:
        st.info("💡 Lütfen yukarıdaki alandan bir görsel yükleyin.")


def page_dashboard():
    st.title("📊 Dinamik Gösterge Paneli & Canlı Harita")
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tespit Edilen Nesne", "5 Adet", "+2")
    m2.metric("Risk Oranı", "%80", "-5%")
    m3.metric("Model Doğruluğu", "96.1%", "+1.2%")
    m4.metric("Yanıt Süresi", "0.9 sn", "-0.3 sn")
    st.divider()

    st.subheader("📈 Performans Grafikleri")
    chart_data = {"Kategori": ["YOLO", "LLM", "UI", "Veri"], "Değer": [92, 85, 100, 80]}
    df_dynamic = pd.DataFrame(chart_data)
    
    c1, c2 = st.columns(2)
    with c1:
        fig_bar = px.bar(df_dynamic, x="Kategori", y="Değer", color="Kategori", text_auto=True, template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        fig_pie = px.pie(df_dynamic, names="Kategori", values="Değer", hole=0.4, template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
    st.divider()

    st.subheader("🗺️ Canlı Lokasyon Haritası")
    st.map(pd.DataFrame({'lat': [41.0253, 41.0822, 41.0082], 'lon': [28.8903, 29.0522, 28.9784]}), zoom=10)


def page_csv_tool():
    st.title("📁 CSV Veri Temizleme & İndirme Paneli")
    st.divider()
    uploaded_csv = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])

    if uploaded_csv is not None:
        df_raw = pd.read_csv(uploaded_csv)
        st.dataframe(df_raw.head(), use_container_width=True)
        if st.button("🧹 Boş Satırları Temizle"):
            df_clean = df_raw.dropna()
            st.success(f"Temizleme tamamlandı! Yeni satır sayısı: {len(df_clean)}")
            st.download_button("📥 Temizlenmiş CSV'yi İndir", df_clean.to_csv(index=False).encode('utf-8'), "LINTu_Cleaned.csv", "text/csv")


def page_team():
    st.title("⚡ LİNTu Yapay Zeka Takımı")
    st.divider()
    ekip_data = {
        "İsim": ["Yusuf", "Tolga", "Senih", "Tunahan", "Alperen", "Cemal & Tuna"],
        "Sorumlu Olduğu Alan": ["Görüntü İşleme", "LLM / Çeviri", "LLM", "Arayüz", "Arayüz", "Veri"],
        "Araçlar": ["YOLOv8", "Gemini API", "Prompt Eng.", "Streamlit", "Plotly", "Pandas"]
    }
    st.table(pd.DataFrame(ekip_data))


# ==========================================
# ANA ÇALIŞTIRICI (MAIN)
# ==========================================
def main():
    st.set_page_config(page_title="LİNTu Team - Central AI Dashboard", page_icon="⚡", layout="wide")
    
    secili_sayfa = render_sidebar()
    
    if secili_sayfa == "🖼️ Görsel Ön İşleme & AI Analiz":
        page_vision_analysis()
    elif secili_sayfa == "📊 Dinamik Grafik & Harita Paneli":
        page_dashboard()
    elif secili_sayfa == "📁 CSV Veri İşleme & İndirme":
        page_csv_tool()
    elif secili_sayfa == "👥 Takım Kadrosu & İstasyonlar":
        page_team()

if __name__ == "__main__":
    main()
