# pages/3_🤖_AI_Chatbot.py

import streamlit as st
import google.generativeai as genai

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="PIKOPI - AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot PIKOPI")
st.caption("Tanya apa saja tentang dunia kopi!")

# --- FUNGSI UNTUK MENGAMBIL API KEY DENGAN AMAN ---
def get_google_api_key():
    """
    Mengambil API key dari st.secrets jika di-deploy.
    Fungsi ini akan memudahkan pengujian lokal juga.
    """
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        return None

# --- Konfigurasi API Key ---
GOOGLE_API_KEY = get_google_api_key()

if not GOOGLE_API_KEY:
    st.error("Google API Key tidak dikonfigurasi. Mohon atur di secrets management Streamlit.")
    st.info("Jika Anda menjalankan ini secara lokal, buat file .streamlit/secrets.toml")
    st.stop()

# Konfigurasi model Generative AI
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Konfigurasi API gagal. Periksa kembali API Key Anda di secrets. Error: {e}")
    st.stop()

# --- Logika Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Tampilkan history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dari pengguna
if prompt := st.chat_input("Apa yang ingin Anda ketahui tentang kopi?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("PIKOPI sedang meracik jawaban..."):
            response = st.session_state.chat_session.send_message(prompt)
            assistant_response = response.text
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    except Exception as e:
        st.error(f"Maaf, terjadi kesalahan: {e}")