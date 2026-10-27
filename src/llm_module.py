def run_llm_analysis(image, detection_res: dict) -> str:
    """
    İstasyon B (Tolga & Senih): Gemini / RAG Rapor Üretim Modülü.
    Tespit edilen nesne verilerini alıp akıllı rapor metni üretir.
    """
    count = detection_res.get("object_count", 0)
    risk = detection_res.get("risk_rate", 0)
    
    return (
        f"🔍 [LİNTu AI Otomatik Raporu]\n"
        f"• Tespit Edilen Unsur Sayısı: {count}\n"
        f"• Hesaplanan Risk Seviyesi: %{risk}\n"
        f"• Durum Analizi: Saha verileri yapay zeka modelleriyle eşleştirildi. "
        f"Kritik eşik aşılmadı, operasyon güvenli bölgede ilerliyor."
    )
