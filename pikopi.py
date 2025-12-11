import streamlit as st

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="PIKOPI - Platform Kopi Cerdas",
    page_icon="☕️",
    layout="wide",
    initial_sidebar_state="collapsed", 
    menu_items={
        'About': "# PIKOPI: AI Coffee Assistant"
    }
)

# --- 2. CUSTOM CSS (Tema PIKOPI) ---
st.markdown("""
<style>
    /* Background & Text Colors sudah diatur di config.toml */
    
    /* Hero Title */
    .hero-title {
        font-size: 3.5em;
        font-weight: 800;
        color: #D4A373;
        text-align: center;
        margin-bottom: 10px;
        font-family: 'Helvetica', sans-serif;
    }
    .hero-subtitle {
        font-size: 1.2em;
        color: #E6D7C3;
        text-align: center;
        margin-bottom: 40px;
    }

    /* Card Style untuk Menu */
    .menu-card {
        background-color: #2C221E;
        border: 1px solid #5C4033;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .menu-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
        border-color: #D4A373;
    }
    
    /* Icon di dalam Card */
    .card-icon {
        font-size: 3em;
        margin-bottom: 15px;
    }
    
    /* Judul Card */
    .card-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #D4A373;
        margin-bottom: 10px;
    }
    
    /* Deskripsi Card */
    .card-desc {
        font-size: 0.9em;
        color: #B09E80;
        margin-bottom: 20px;
    }

    /* Tombol di dalam Card */
    div.stButton > button {
        background-color: transparent;
        border: 2px solid #D4A373;
        color: #D4A373;
        border-radius: 25px;
        padding: 8px 20px;
        font-weight: bold;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #D4A373;
        color: #2C221E;
        border-color: #D4A373;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #5C4033;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HERO SECTION ---
st.markdown('<div class="hero-title">PIKOPI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Platform Cerdas untuk Penjelajah Kopi Indonesia 🇮🇩<br>Powered by Artificial Intelligence</div>', unsafe_allow_html=True)

st.markdown("---")

# --- 4. MENU NAVIGATION (GRID LAYOUT) ---
# Baris 1: Fitur Utama (Chatbot & Personalization)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="menu-card">
        <div class="card-icon">💬</div>
        <div class="card-title">AI Barista Chatbot</div>
        <div class="card-desc">
            Tanya apa saja soal kopi, cari resep seduh, atau temukan cafe terdekat di kotamu.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Mulai Chatting", key="btn_chat", use_container_width=True):
        st.switch_page("pages/chatbot.py")

with col2:
    st.markdown("""
    <div class="menu-card">
        <div class="card-icon">💡</div>
        <div class="card-title">Personalized Coffee</div>
        <div class="card-desc">
            Bingung pilih biji kopi? Ceritakan seleramu, AI akan mencarikan jodoh kopi terbaik.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Cari Rekomendasi", key="btn_personal", use_container_width=True):
        st.switch_page("pages/personalization.py")

st.markdown("<br>", unsafe_allow_html=True) # Spacer

# Baris 2: Fitur Analisis (Classification & Prediction)
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="menu-card">
        <div class="card-icon">🖼️</div>
        <div class="card-title">Bean Classifier</div>
        <div class="card-desc">
            Scan foto biji kopi untuk mendeteksi jenis (Peaberry, Longberry) atau cacat (Defect).
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Scan Biji Kopi", key="btn_class", use_container_width=True):
        st.switch_page("pages/clasification.py")

with col4:
    st.markdown("""
    <div class="menu-card">
        <div class="card-icon">📈</div>
        <div class="card-title">Score Predictor</div>
        <div class="card-desc">
            Prediksi skor cupping (kualitas) kopi berdasarkan data fisik atau sensoris.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Prediksi Skor", key="btn_pred", use_container_width=True):
        st.switch_page("pages/prediction.py")

# --- 5. FOOTER ---
st.markdown("---")
st.markdown('<div class="footer">© 2025 PIKOPI Project. Dibuat dengan ❤️ dan ☕ untuk Studi Independen.</div>', unsafe_allow_html=True)