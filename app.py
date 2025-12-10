import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartFinance AI", layout="wide")

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.title("💸 SmartFinance AI — Pengatur Keuangan Otomatis")
st.write("Aplikasi cerdas untuk membantu mengatur dan menganalisis pengeluaran harian.")

# -------------------------------------------------------
# UPLOAD FILE
# -------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload file transaksi (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Transaksi")
    st.dataframe(df)

    required_cols = ["tanggal", "kategori", "jumlah"]
    if not all(col in df.columns for col in required_cols):
        st.error("CSV harus memiliki kolom: tanggal, kategori, jumlah")
        st.stop()

    # Konversi ke datetime
    df["tanggal"] = pd.to_datetime(df["tanggal"])

    # -------------------------------------------------------
    # STATISTIK
    # -------------------------------------------------------
    st.subheader("📊 Statistik Pengeluaran")

    total = df["jumlah"].sum()
    st.metric("Total Pengeluaran", f"Rp {total:,.0f}")

    # Per kategori
    kategori_sum = df.groupby("kategori")["jumlah"].sum()

    st.write("### Pengeluaran per Kategori")
    st.bar_chart(kategori_sum)

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(kategori_sum, labels=kategori_sum.index, autopct='%1.1f%%')
    ax.axis("equal")
    st.pyplot(fig)

    # -------------------------------------------------------
    # AI REKOMENDASI
    # -------------------------------------------------------
    st.subheader("🤖 Rekomendasi AI")

    rekomendasi = []

    makanan = df[df["kategori"]=="makanan"]["jumlah"].sum() if "makanan" in df["kategori"].unique() else 0
    hiburan = df[df["kategori"]=="hiburan"]["jumlah"].sum() if "hiburan" in df["kategori"].unique() else 0

    if makanan > total * 0.35:
        rekomendasi.append("Pengeluaran makanan terlalu tinggi. Coba atur menu atau masak sendiri.")

    if hiburan > total * 0.20:
        rekomendasi.append("Biaya hiburan cukup besar. Kurangi kegiatan tidak mendesak.")

    if total > 3000000:
        rekomendasi.append("Total pengeluaran bulan ini cukup tinggi. Pertimbangkan membuat anggaran mingguan.")

    if len(rekomendasi) == 0:
        st.success("Keuangan stabil! Tidak ditemukan pengeluaran berlebihan.")
    else:
        for r in rekomendasi:
            st.warning(r)

    # -------------------------------------------------------
    # PREDIKSI PENGELUARAN (AI)
    # -------------------------------------------------------
    st.subheader("📈 Prediksi Pengeluaran Bulan Depan (AI)")

    df["bulan"] = df["tanggal"].dt.month
    monthly = df.groupby("bulan")["jumlah"].sum().reset_index()

    X = monthly[["bulan"]]
    y = monthly["jumlah"]

    model = LinearRegression()
    model.fit(X, y)

    next_month = np.array([[monthly["bulan"].max() + 1]])
    pred = model.predict(next_month)[0]

    st.info(f"Prediksi pengeluaran bulan depan: **Rp {pred:,.0f}**")

    # -------------------------------------------------------
    # RINGKASAN OTOMATIS
    # -------------------------------------------------------
    st.subheader("📝 Ringkasan Laporan Otomatis")

    summary = f"""
    **Ringkasan Pengeluaran:**
    - Total Pengeluaran: Rp {total:,.0f}
    - Pengeluaran Terbesar: {kategori_sum.idxmax()} (Rp {kategori_sum.max():,.0f})
    - Bulan dengan pengeluaran tertinggi: {monthly.loc[monthly['jumlah'].idxmax(), 'bulan']}
    - Prediksi Pengeluaran Bulan Depan: Rp {pred:,.0f}

    **Catatan AI:**
    {"; ".join(rekomendasi) if rekomendasi else "Keuangan stabil dan sehat!"}
    """

    st.write(summary)

else:
    st.info("Silakan upload file CSV untuk mulai menganalisis.")
