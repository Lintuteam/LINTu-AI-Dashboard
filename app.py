"""LİNTu Team · Merkezi AI Dashboard (İstasyon C: Tunahan & Alperen)

Gün 6 yapısı: modüller arası sınır net, model henüz bağlı değilken uygulama
sahte sonuç göstermez, "bekleniyor" der.
"""
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

# --- Entegrasyon noktaları (modüller gelince yorumları aç, MODEL_READY = True yap) ---
# from src.vision_module import detect_objects   # Yusuf  -> list[dict]: label, conf, box
# from src.llm_module import analyze_with_ai     # Tolga & Senih

MODEL_READY = False
DATA_PATH = Path(__file__).parent / "data" / "ornek.csv"


# ==========================================
# ENTEGRASYON KATMANI (İstasyon A / B bağlantı noktaları)
# ==========================================
def run_detection(image):
    """Yusuf'un detect_objects(image) çıktısını döndürür.

    Beklenen biçim: [{"label": str, "conf": float (0-1), "box": [x1, y1, x2, y2]}]
    Model bağlı değilse None döner (sahte değer üretilmez).
    """
    if not MODEL_READY:
        return None
    return detect_objects(image)  # noqa: F821  (import yukarıda yorumdan çıkarılınca çalışır)


def make_report(res):
    """Tespit sonucundan düz metin rapor üretir. Şimdilik yalnızca eldeki sayılardan.

    TODO: analyze_with_ai(...) (Gemini) bağlanınca burası onunla değişecek.
    """
    if res is None:
        return None
    if not res:
        return "Görselde nesne tespit edilmedi."
    sayim = {}
    for d in res:
        sayim[d["label"]] = sayim.get(d["label"], 0) + 1
    ort_guven = sum(d["conf"] for d in res) / len(res) * 100
    dagilim = ", ".join(f"{ad}: {adet}" for ad, adet in sayim.items())
    return (
        f"Toplam {len(res)} nesne tespit edildi.\n"
        f"Ortalama güven skoru: %{ort_guven:.1f}\n"
        f"Dağılım: {dagilim}"
    )


# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================
def load_image(uploaded):
    """Yüklenen dosyayı güvenle açar: EXIF yönünü düzeltir, RGB'ye çevirir."""
    try:
        img = Image.open(uploaded)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def apply_filter(img, name):
    if name == "Siyah-Beyaz (Grayscale)":
        return ImageOps.grayscale(img)
    if name == "Kenar Tespiti (Edge Detection)":
        return img.filter(ImageFilter.FIND_EDGES)
    if name == "Bulanıklaştırma (Blur)":
        return img.filter(ImageFilter.BLUR)
    return img


def read_csv_safely(uploaded):
    """Türkçe Excel çıktıları için: utf-8 / cp1254 ve ; , ayraçlarını dener."""
    raw = uploaded.getvalue()
    text = None
    for enc in ("utf-8-sig", "cp1254"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("Dosya kodlaması çözülemedi.")
    try:
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        return pd.read_csv(io.StringIO(text))


@st.cache_data
def load_sample_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(
        {"kategori": ["Market", "Ulaşım", "Fatura", "Yemek"], "tutar": [1250, 450, 1800, 640]}
    )


# ==========================================
# UI FONKSİYONLARI (İSTASYON C)
# ==========================================
def render_sidebar():
    st.sidebar.title("⚡ LİNTu AI Dashboard")
    st.sidebar.caption("İstasyon C: Tunahan & Alperen")
    secim = st.sidebar.radio("Modül Seçimi:", list(PAGES))
    st.sidebar.divider()
    if MODEL_READY:
        st.sidebar.success("🟢 Model bağlı")
    else:
        st.sidebar.warning("🟡 Demo modu · YOLO / LLM henüz bağlı değil")
    return secim


def render_result(res):
    """Sonuç kutusu. res None ise 'bekleniyor' durumunu gösterir."""
    m1, m2 = st.columns(2)
    if res is None:
        m1.metric("Tespit Edilen Nesne", "—")
        m2.metric("Ortalama Güven", "—")
        st.info("⏳ Sonuç bekleniyor. Fotoğrafı yükleyip 'Analiz et'e bas; "
                "model bağlı değilse sonuç burada görünmez.")
        return
    ort = (sum(d["conf"] for d in res) / len(res) * 100) if res else 0.0
    m1.metric("Tespit Edilen Nesne", f"{len(res)} Adet")
    m2.metric("Ortalama Güven", f"%{ort:.1f}")


def page_vision_analysis():
    st.title("🖼️ Görsel Ön İşleme & Yapay Zeka Analiz Paneli")
    st.caption("Görsel Filtreleme, YOLO Nesne Tanıma ve LLM Otomatik Raporlama")
    st.divider()

    uploaded = st.file_uploader("Analiz için bir görsel yükleyin (JPG, PNG)",
                                type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.session_state.pop("analysis", None)
        st.info("💡 Lütfen yukarıdaki alandan bir görsel yükleyin.")
        return

    image = load_image(uploaded)
    if image is None:
        st.error("Görsel okunamadı. Dosya bozuk olabilir, başka bir JPG/PNG dene.")
        return

    col_sol, col_sag = st.columns(2)

    with col_sol:
        st.subheader("📸 Görsel Ön İşleme (Filtreler)")
        filtre = st.selectbox(
            "Model Öncesi Görsel Filtresi:",
            ["Orijinal", "Siyah-Beyaz (Grayscale)", "Kenar Tespiti (Edge Detection)",
             "Bulanıklaştırma (Blur)"],
        )
        processed = apply_filter(image, filtre)
        st.image(processed, caption=f"Uygulanan Filtre: {filtre}")

    with col_sag:
        st.subheader("🤖 Yapay Zeka Model Çıktıları")
        key = (uploaded.name, uploaded.size, filtre)

        if st.button("🔍 Analiz et"):
            try:
                st.session_state["analysis"] = {"key": key, "res": run_detection(processed)}
            except Exception as e:  # model hatası arayüzü çökertmesin
                st.error(f"Analiz sırasında hata: {e}")

        analysis = st.session_state.get("analysis")
        res = analysis["res"] if analysis and analysis["key"] == key else None
        render_result(res)

        st.divider()
        st.markdown("##### 📝 Otomatik Rapor")
        report = make_report(res)
        if report:
            st.text_area("Model Analiz Raporu:", value=report, height=140)
            st.download_button("📥 Raporu İndir (.txt)", report.encode("utf-8"),
                               "LINTu_rapor.txt", "text/plain")
        else:
            st.caption("Rapor, analiz sonucu geldikten sonra burada oluşacak.")


def page_dashboard():
    st.title("📊 Dinamik Gösterge Paneli & Lokasyon Haritası")
    st.caption("Örnek veri: data/ornek.csv")
    st.divider()

    df = load_sample_data()
    ozet = (df.groupby("kategori", as_index=False)["tutar"].sum()
              .sort_values("tutar", ascending=False))
    toplam = ozet["tutar"].sum()
    en_buyuk = ozet.iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Tutar", f"{toplam:,.0f} ₺".replace(",", "."))
    m2.metric("Kategori Sayısı", len(ozet))
    m3.metric("Kayıt Sayısı", len(df))
    m4.metric("En Büyük Kategori", en_buyuk["kategori"],
              f"%{en_buyuk['tutar'] / toplam * 100:.0f}", delta_color="off")
    st.divider()

    st.subheader("📈 Kategori Grafikleri")
    palet = px.colors.qualitative.Plotly
    renk = {k: palet[i % len(palet)] for i, k in enumerate(sorted(ozet["kategori"]))}

    c1, c2 = st.columns(2)
    with c1:
        fig_bar = px.bar(ozet, x="kategori", y="tutar", color="kategori",
                         color_discrete_map=renk, text_auto=True, template="plotly_dark")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar)
    with c2:
        fig_pie = px.pie(ozet, names="kategori", values="tutar", color="kategori",
                         color_discrete_map=renk, hole=0.4, template="plotly_dark")
        st.plotly_chart(fig_pie)

    with st.expander("Veriyi göster"):
        st.dataframe(df)
    st.divider()

    st.subheader("🗺️ Lokasyon Haritası (örnek noktalar)")
    st.map(pd.DataFrame({"lat": [41.0253, 41.0822, 41.0082],
                         "lon": [28.8903, 29.0522, 28.9784]}), zoom=10)


def page_csv_tool():
    st.title("📁 CSV Veri Temizleme & İndirme Paneli")
    st.divider()
    uploaded = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])
    if uploaded is None:
        st.session_state.pop("csv_state", None)
        return

    sig = (uploaded.name, uploaded.size)
    state = st.session_state.get("csv_state")
    if not state or state["sig"] != sig:  # yeni dosya -> eski sonucu sıfırla
        try:
            state = {"sig": sig, "raw": read_csv_safely(uploaded), "clean": None}
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")
            return
        st.session_state["csv_state"] = state

    df_raw = state["raw"]
    st.caption(f"{len(df_raw)} satır · {len(df_raw.columns)} sütun")
    st.dataframe(df_raw.head())

    sil_bos = st.checkbox("Boş satırları sil", value=True)
    sil_tekrar = st.checkbox("Tekrar eden satırları sil", value=False)

    if st.button("🧹 Temizle"):
        df = df_raw
        if sil_bos:
            df = df.dropna(how="all")
        if sil_tekrar:
            df = df.drop_duplicates()
        state["clean"] = df

    # Buton bloğunun DIŞINDA: indir'e basınca sonuç kaybolmaz
    if state["clean"] is not None:
        df_clean = state["clean"]
        st.success(f"Temizleme tamamlandı! {len(df_raw)} → {len(df_clean)} satır")
        st.download_button("📥 Temizlenmiş CSV'yi İndir",
                           df_clean.to_csv(index=False).encode("utf-8-sig"),
                           "LINTu_Cleaned.csv", "text/csv")


def page_team():
    st.title("⚡ LİNTu Yapay Zeka Takımı")
    st.caption("Ay 1 (16 Ekim – 15 Kasım) rotasyonuna göre")
    st.divider()
    ekip = pd.DataFrame({
        "İsim": ["Yusuf", "Tolga", "Senih", "Tunahan", "Alperen", "Cemal"],
        "Ay 1 İstasyonu": [
            "A · Görüntü İşleme",
            "B · LLM & Ajanlar",
            "B · LLM & Ajanlar",
            "C · Arayüz + D · Veri",
            "C · Arayüz & Harita",
            "D · Veri & Sunum",
        ],
        "Araçlar": [
            "YOLOv8, OpenCV",
            "Gemini API",
            "Gemini API, Prompt Eng.",
            "Streamlit, Pandas",
            "Streamlit, Plotly",
            "Pandas, Canva",
        ],
    })
    st.table(ekip)


PAGES = {
    "🖼️ Görsel Ön İşleme & AI Analiz": page_vision_analysis,
    "📊 Dinamik Grafik & Harita Paneli": page_dashboard,
    "📁 CSV Veri İşleme & İndirme": page_csv_tool,
    "👥 Takım Kadrosu & İstasyonlar": page_team,
}


# ==========================================
# ANA ÇALIŞTIRICI
# ==========================================
def main():
    st.set_page_config(page_title="LİNTu Team - Central AI Dashboard",
                       page_icon="⚡", layout="wide")
    secim = render_sidebar()
    PAGES[secim]()


if __name__ == "__main__":
    main()
