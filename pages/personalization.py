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

# --- 3. KONFIGURASI HALAMAN & CSS ---
st.set_page_config(
    page_title="Barista AI Rekomen",
    page_icon="☕",
    layout="centered"
)

# Custom CSS Tema PIKOPI
st.markdown("""
<style>
    /* Judul H1 */
    h1 {
        color: #D4A373 !important;
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
    }
    
    /* Subheader */
    h3 {
        color: #E6D7C3 !important;
    }

    /* Tombol Utama */
    div.stButton > button:first-child {
        background-color: #D4A373;
        color: #2C221E;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #E9C46A;
        color: #1E1614;
        transform: scale(1.02);
    }

    /* Chat Bubble User */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #3E322C; 
        color: #E6D7C3;
        border-radius: 15px;
    }
    
    /* Chat Bubble Assistant */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: transparent;
        border: 1px solid #D4A373;
        border-radius: 15px;
    }

    /* Slider Color */
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
        background-color: #D4A373 !important;
    }
    
    /* Card Container */
    .input-card {
        background-color: #2C221E;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #5C4033;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def reset_conversation():
    st.session_state.messages = []
    st.session_state.recommendation_done = False

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2935/2935413.png", width=80)
    st.title("Pengaturan")
    st.info("Ingin mencari profil rasa kopi yang berbeda?")
    if st.button("🔄 Mulai Baru", type="primary"):
        reset_conversation()
        st.rerun()

# --- HEADER ---
st.title("☕ Personal Barista AI")
st.markdown("Ceritakan rasa kopi impianmu, biarkan AI mencarikan biji kopi yang paling pas buat lidahmu.")
st.markdown("---")

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

# --- UI INPUT RASA (Hanya muncul jika belum ada rekomendasi) ---
if not st.session_state.recommendation_done:
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🎚️ Tentukan Profil Rasa")
        
        rasa_options = [
            "Fruity (Buah-buahan)", "Acidity (Asam Segar)", "Manis/Sweet", 
            "Pahit/Bitter", "Strong/Bold", "Nutty (Kacang)", 
            "Chocolate", "Caramel", "Spices (Rempah)", "Floral (Bunga)", "Creamy"
        ]
        
        selected_rasa = st.multiselect(
            "Pilih karakteristik rasa yang kamu cari:", 
            rasa_options, 
            placeholder="Klik di sini untuk memilih rasa..."
        )

        prompt_requirements = [] 
        
        if selected_rasa:
            st.markdown("<br>", unsafe_allow_html=True)
            st.write("**Atur Intensitas Rasa:**")
            
            for rasa in selected_rasa:
                clean_rasa_name = rasa.split(" (")[0] 
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{clean_rasa_name}**")
                with col2:
                    level = st.slider(f"Level {clean_rasa_name}", 1, 10, 5, key=f"slider_{clean_rasa_name}", label_visibility="collapsed")
                
                if level <= 3: desc = "halus/tipis (hint of)"
                elif level <= 7: desc = "sedang/seimbang (balanced)"
                else: desc = "sangat kuat/dominan (bold/intense)"
                
                prompt_requirements.append(f"- Rasa {clean_rasa_name}: {desc} (Level {level}/10)")

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔍 Temukan Kopi Cocok", use_container_width=True):
            if not selected_rasa:
                st.warning("⚠️ Mohon pilih minimal satu karakteristik rasa di atas.")
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
        
        st.markdown('</div>', unsafe_allow_html=True)


# --- UI HASIL & CHAT ---
if st.session_state.recommendation_done:
    st.subheader("✨ Rekomendasi Barista")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Tanya detail lain tentang kopi ini..."):
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