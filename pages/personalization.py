import streamlit as st
import sys
import os

# --- 1. SETUP PATH IMPORT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- 2. IMPORT DARI MODULES ---
try:
    from modules.personalization.model import get_rag_chain
except ImportError as e:
    st.error(f"Gagal import modul: {e}")
    st.stop()

# --- 3. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Barista AI Rekomen",
    page_icon="☕",
    layout="centered"
)

# --- CSS CUSTOM (Tema PIKOPI) ---
st.markdown("""
<style>
    h1 { color: #D4A373 !important; }
    div.stButton > button:first-child {
        background-color: #D4A373;
        color: #2C221E;
        font-weight: bold;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

def reset_conversation():
    st.session_state.messages = []
    st.session_state.recommendation_done = False

with st.sidebar:
    st.header("Pengaturan")
    st.write("Ingin cari kopi lain?")
    if st.button("🔄 Mulai Percakapan Baru", type="primary"):
        reset_conversation()
        st.rerun()

st.title("☕ Asisten Barista AI")
st.write("Ceritakan rasa kopi yang kamu inginkan, aku akan carikan yang pas!")

# Load Chain (Cached)
@st.cache_resource
def load_chain():
    return get_rag_chain()

chain = load_chain()

if not chain:
    st.error("Gagal memuat AI Chain. Cek koneksi database.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "recommendation_done" not in st.session_state:
    st.session_state.recommendation_done = False

# --- UI INPUT RASA ---
with st.container():
    st.subheader("1. Tentukan Profil Rasa")
    
    rasa_options = [
        "Fruity (Buah-buahan)", 
        "Acidity (Asam Segar)", 
        "Manis/Sweet", 
        "Pahit/Bitter", 
        "Strong/Bold", 
        "Nutty (Kacang)", 
        "Chocolate", 
        "Caramel", 
        "Spices (Rempah)", 
        "Floral (Bunga)", 
        "Creamy"
    ]
    selected_rasa = st.multiselect("Pilih karakteristik rasa yang kamu cari:", rasa_options, placeholder="Pilih rasa...")

    prompt_requirements = [] 
    
    if selected_rasa:
        st.markdown("---")
        st.write("🎚️ **Atur Intensitas Rasa:**")
        
        for rasa in selected_rasa:
            clean_rasa_name = rasa.split(" (")[0] 
            
            level = st.slider(f"Seberapa kuat rasa **{clean_rasa_name}**?", 1, 10, 5, key=f"slider_{clean_rasa_name}")
            
            if level <= 3:
                desc = "halus/tipis (hint of)"
            elif level <= 7:
                desc = "sedang/seimbang (balanced)"
            else:
                desc = "sangat kuat/dominan (bold/intense)"
            
            prompt_requirements.append(f"- Rasa {clean_rasa_name}: {desc} (Level {level}/10)")

    if st.button("🔍 Temukan Kopi"):
        if not selected_rasa:
            st.warning("Mohon pilih minimal satu karakteristik rasa.")
        else:
            req_text = "\n".join(prompt_requirements)
            
            final_user_query = f"""
            Tolong carikan kopi dengan spesifikasi rasa berikut:
            
            {req_text}
            
            Abaikan asal daerah/jenis biji, fokus hanya pada profil rasa yang diminta di atas.
            Berikan rekomendasi yang paling mendekati kombinasi tersebut.
            """
            
            st.session_state.messages = [] 
            
            display_message = f"Aku cari kopi dengan profil: **{', '.join(selected_rasa)}**."
            st.session_state.messages.append({"role": "user", "content": display_message})
            
            with st.spinner("Barista sedang mencicipi data..."):
                try:
                    response = chain.invoke(final_user_query)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.recommendation_done = True 
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Gagal mencari rekomendasi: {e}")


# --- UI HASIL & CHAT ---
if st.session_state.recommendation_done:
    st.divider()
    st.subheader("2. Rekomendasi & Diskusi")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Tanya detail lain"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mengetik..."):
                try:
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")

elif not st.session_state.recommendation_done and st.session_state.messages:
    st.session_state.messages = []