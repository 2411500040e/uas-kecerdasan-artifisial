import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartFinance AI", layout="wide")

# ================= CUSTOM CSS (PINK THEME) =================
pink_css = """
<style>
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

header, .css-18e3th9 {
    background-color: #ffdee9 !important;
}

.main {
    background-color: #fff1f7 !important;
}

h1, h2, h3, h4 {
    color: #d63384 !important;
}

.stButton>button {
    background-color: #ff66a3 !important;
    color: white;
    border-radius: 8px;
    border: none;
}

.stMetric {
    background: #ffb3cc !important;
    padding: 20px;
    border-radius: 15px;
}

.block-container {
    padding-top: 2rem;
}

.upload-box {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 2px dashed #ff8fb8;
}
</style>
"""
st.markdown(pink_css, unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div style="padding:20px; background:#ffb6d9; border-radius:10px;">
<h1>💸 SmartFinance AI — Pink Edition</h1>
<p>Aplikasi pintar untuk analisis keuangan dengan tampilan pink yang manis.</p>
</div>
""", unsafe_allow_html=True)

# ================= UPLOAD CSV =================
st.subheader("📤 Upload Data Keuangan")
uploaded_file = st.file_uploader("Upload file CSV kamu:", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("File berhasil diupload!")
    st.dataframe(df)

    required_cols = ["tanggal", "kategori", "jumlah"]
    if not all(col in df.columns for col in required_cols):
        st.error("CSV harus memiliki kolom: tanggal, kategori, jumlah")
        st.stop()

    df["tanggal"] = pd.to_datetime(df["tanggal"])

    # ================= DASHBOARD =================
    st.subheader("🎀 Dashboard Keuangan")

    col1, col2 = st.columns(2)

    total = df["jumlah"].sum()
    kategori_sum = df.groupby("kategori")["jumlah"].sum()

    with col1:
        st.metric("Total Pengeluaran", f"Rp {total:,.0f}")

    with col2:
        st.metric("Kategori Terbesar", kategori_sum.idxmax())

    st.write("### 📊 Grafik Pengeluaran per Kategori (Pink Theme)")
    st.bar_chart(kategori_sum)

    fig, ax = plt.subplots()
    ax.pie(kategori_sum, labels=kategori_sum.index, autopct='%1.1f%%')
    ax.axis("equal")
    st.pyplot(fig)

    # ================= REKOMENDASI AI =================
    st.subheader("💗 Rekomendasi AI")
    rekom = []

    makanan = kategori_sum.get("makanan", 0)
    hiburan = kategori_sum.get("hiburan", 0)

    if makanan > total * 0.35:
        rekom.append("Pengeluaran makanan cukup tinggi, coba lebih hemat ya 💕")

    if hiburan > total * 0.20:
        rekom.append("Hiburan kamu cukup banyak, coba kurangi sedikit 💗")

    if total > 3000000:
        rekom.append("Total pengeluaran cukup besar bulan ini, hati-hati ya 💞")

    if rekom:
        for r in rekom:
            st.warning(r)
    else:
        st.success("Keuangan kamu aman! Tetap pertahankan ya 💖")

    # ================= PREDIKSI AI =================
    st.subheader("📈 Prediksi Pengeluaran (AI)")

    df["bulan"] = df["tanggal"].dt.month
    monthly = df.groupby("bulan")["jumlah"].sum().reset_index()

    model = LinearRegression()
    model.fit(monthly[["bulan"]], monthly["jumlah"])

    next_month = monthly["bulan"].max() + 1
    pred = model.predict([[next_month]])[0]

    st.info(f"💞 Prediksi bulan depan: Rp {pred:,.0f}")

    # ================= RINGKASAN =================
    st.subheader("📝 Ringkasan Laporan")

    st.write(f"""
    **Ringkasan Pengeluaran:**
    - Total: Rp {total:,.0f}
    - Kategori Terbesar: {kategori_sum.idxmax()} (Rp {kategori_sum.max():,.0f})
    - Prediksi Bulan Depan: Rp {pred:,.0f}

    **Rekomendasi AI:**
    {", ".join(rekom) if rekom else "Keuangan kamu dalam kondisi baik 💗"}
    """)

else:
    st.info("Silakan upload file CSV untuk mulai analisis 🎀")
