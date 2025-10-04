# PIKOPI.py

import streamlit as st

# Konfigurasi halaman utama
st.set_page_config(
    page_title="PIKOPI - Dashboard Utama",
    page_icon="☕️",
    layout="wide"
)

# Judul dan deskripsi dashboard
st.title("PIKOPI: Platform Cerdas Kopi Anda ☕️")
st.markdown("Selamat datang di PIKOPI! Aplikasi ini dirancang untuk membantu Anda menjelajahi dunia kopi dengan bantuan kecerdasan buatan. Silakan pilih salah satu fitur di bawah ini untuk memulai.")
st.divider()

# Membuat layout kolom untuk tombol agar terlihat rapi
col1, col2 = st.columns(2)

with col1:
    st.subheader("Analisis & Prediksi")
    st.markdown("Gunakan model AI kami untuk menganalisis biji kopi Anda atau memprediksi metrik penting.")
    
    # Tombol untuk navigasi ke halaman Klasifikasi
    if st.button("🖼️ Klasifikasi Biji Kopi", use_container_width=True):
        st.switch_page("pages/clasification.py")
        
    # Tombol untuk navigasi ke halaman Prediksi
    if st.button("📈 Prediksi Harga / Skor Kopi", use_container_width=True):
        st.switch_page("pages/prediction.py")

with col2:
    st.subheader("Asisten & Rekomendasi")
    st.markdown("Berinteraksi dengan asisten AI kami atau dapatkan rekomendasi kopi yang dipersonalisasi.")
    
    # Tombol untuk navigasi ke halaman Chatbot
    if st.button("💬 AI Chatbot", use_container_width=True):
        st.switch_page("pages/chatbot.py")

    # Tombol untuk navigasi ke halaman Saran
    if st.button("💡 Saran Kopi Sesuai Selera", use_container_width=True):
        st.switch_page("pages/personalization.py")

st.divider()
st.markdown("Dibuat sebagai Proyek Studi Independen.")