import streamlit as st
import google.generativeai as genai
import os

# --- Konfigurasi Halaman dan Judul ---
st.set_page_config(
    page_title="Kopi.AI Chatbot",
    page_icon="☕️",
    layout="centered"
)

st.title("Kopi.AI Chatbot ☕️")
st.caption("Tahap 1: Asisten Kopi Cerdas Berbasis Gemini 1.5 Flash")

# --- Konfigurasi API Key di Sidebar ---
with st.sidebar:
    st.header("Konfigurasi")
    # Cara yang lebih aman untuk deployment: gunakan st.secrets
    # Tapi untuk pengembangan lokal, text_input sudah cukup.
    google_api_key = st.text_input(
        "Masukkan Google API Key Anda:", 
        type="password",
        help="Dapatkan API key Anda dari Google AI Studio."
    )
    if not google_api_key:
        st.info("Mohon masukkan API Key Anda untuk memulai percakapan.")
        st.stop()
    
    # Tombol untuk memulai percakapan baru
    if st.button("Mulai Percakapan Baru"):
        # Hapus history dari session state
        st.session_state.messages = []
        st.session_state.chat = None
        st.rerun()

# Konfigurasi model Generative AI setelah API key dimasukkan
try:
    genai.configure(api_key=google_api_key)
except Exception as e:
    st.error(f"Konfigurasi API gagal. Periksa kembali API Key Anda. Error: {e}")
    st.stop()

# --- Inisialisasi Model dan Session State ---

# Inisialisasi model Gemini
model = genai.GenerativeModel('gemini-2.5-flash')

# Inisialisasi session state untuk menyimpan history percakapan
if "messages" not in st.session_state:
    st.session_state.messages = []

# Inisialisasi objek chat jika belum ada
if "chat" not in st.session_state or st.session_state.chat is None:
    st.session_state.chat = model.start_chat(history=[])


# --- Tampilkan History Percakapan ---
for message in st.session_state.messages:
    # Tampilkan pesan dari history dengan role yang sesuai
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Input dari Pengguna dan Respon dari Model ---
if prompt := st.chat_input("Tanya apa saja tentang kopi..."):
    # 1. Tambahkan dan tampilkan pesan pengguna ke UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Kirim prompt ke model Gemini dan dapatkan responnya
    try:
        with st.spinner("Kopi.AI sedang berpikir... 🤔"):
            # Gunakan objek chat yang sudah ada untuk mengirim pesan
            # Ini akan otomatis menjaga konteks percakapan
            response = st.session_state.chat.send_message(prompt)
        
        # 3. Tambahkan dan tampilkan respon dari model
        assistant_response = response.text
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat berkomunikasi dengan AI: {e}")