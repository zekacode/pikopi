import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px

# --- 1. SETUP PATH IMPORT (FIXED) ---
current_file_path = os.path.abspath(__file__)
pages_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(pages_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- 2. IMPORT DARI MODULES ---
try:
    from modules.prediction.preprocess_single import preprocess_single
    from modules.prediction.model_loader import load_model_and_preprocessor
    from modules.prediction.recommend import quality_category, recommendation_from_category
except ImportError as e:
    st.error(f"Gagal import modul: {e}")
    st.stop()

# --- 3. KONFIGURASI HALAMAN & CSS ---
st.set_page_config(page_title="Prediksi Cupping Score", layout="wide", page_icon="☕")

# Custom CSS untuk Tema Kopi
st.markdown("""
<style>
    /* Warna Background Utama sudah diatur di config.toml, kita atur elemen spesifik */
    
    /* Judul H1 */
    h1 {
        color: #D4A373 !important; /* Warna Emas/Karamel */
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
    }
    
    /* Sub-header H2, H3 */
    h2, h3 {
        color: #E6D7C3 !important; /* Warna Krem */
    }

    /* Tombol Prediksi (Primary Button) */
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

    /* Card Hasil (Container) */
    .result-card {
        background-color: #2C221E;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #D4A373;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Highlight Text */
    .highlight {
        color: #D4A373;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_h1, col_h2 = st.columns([1, 5])
with col_h1:
    st.image("https://cdn-icons-png.flaticon.com/512/2935/2935413.png", width=80) # Ikon Biji Kopi
with col_h2:
    st.title("AI Coffee Cupping Predictor")
    st.markdown("Prediksi kualitas dan skor kopi Anda menggunakan kecerdasan buatan.")

st.markdown("---")

# ============================
# Sidebar: Barista Control Panel
# ============================
with st.sidebar:
    st.header("🎛️ Barista Panel")
    st.info("Masukkan parameter kopi di sini.")
    
    mode = st.selectbox("Pilih Mode Analisa:", ["Fisik", "Akurat"]).lower()
    
    # Tombol Reset dengan styling
    if st.button("🔄 Reset Parameter"):
        st.session_state.features = None
        st.rerun()

# ============================
# Default Features Logic
# ============================
def get_default_features(mode):
    if mode == "fisik":
        return {
            "Altitude": 1200, "Coffee Age": 10, "Moisture Percentage": 11.0,
            "Category One Defects": 1, "Category Two Defects": 2, "Quakers": 1,
            "Processing Method": "Natural / Dry", "Variety": "Blend",
        }
    else:
        return {
            "Uniformity": 10.0, "Clean Cup": 10.0, "Sweetness": 10.0,
            "Overall": 8.0, "Flavor": 8.0, "Aftertaste": 8.0,
            "Balance": 8.0, "Acidity": 8.0, "Aroma": 8.0, "Body": 8.0,
            "Processing Method": "Natural / Dry", "Variety": "Blend",
        }

if "features" not in st.session_state or st.session_state.features is None:
    st.session_state.features = get_default_features(mode)

# ============================
# Input Form (Sidebar)
# ============================
input_data = {}
with st.sidebar:
    st.markdown("### 📝 Input Data")
    for key, val in st.session_state.features.items():
        if isinstance(val, float):
            input_data[key] = st.number_input(key, value=val)
        elif isinstance(val, int):
            input_data[key] = st.number_input(key, value=val, step=1)
        else:
            if key == "Processing Method":
                opts = ["Natural / Dry", "Pulped natural / honey", "Washed / Wet"]
            else:
                opts = ["Blend", "Catuai", "Other", "Red Bourbon,Caturra", "unknown"]
            
            # Safe index finding
            idx = opts.index(val) if val in opts else 0
            input_data[key] = st.selectbox(key, opts, index=idx)

# ============================
# Load Model
# ============================
model, preprocessor = load_model_and_preprocessor(mode)
if model is None or preprocessor is None:
    st.error("❌ Model tidak ditemukan. Pastikan file .pkl ada di folder modules/prediction/models/")
    st.stop()

# ============================
# Main Content: Prediction
# ============================

# Tombol Besar di Tengah
col_space1, col_btn, col_space2 = st.columns([1, 2, 1])
with col_btn:
    predict_btn = st.button("☕ Analisa Kualitas Kopi Sekarang", use_container_width=True)

if predict_btn:
    with st.spinner("Sedang me-roasting data..."):
        try:
            # Preprocess & Predict
            input_df = preprocess_single(input_data, mode=mode)
            X = preprocessor.transform(input_df)
            y_pred = model.predict(X)[0]
            
            kategori = quality_category(y_pred)
            rekomendasi = recommendation_from_category(kategori)

            # --- TAMPILAN HASIL (DASHBOARD STYLE) ---
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Container Hasil
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="text-align: center; margin-bottom: 0;">Hasil Prediksi Cupping Score</h2>
                    <h1 style="text-align: center; font-size: 4em; margin: 0; color: #D4A373;">{y_pred:.2f}</h1>
                    <h3 style="text-align: center; color: #E6D7C3;">Kategori: <span style="color: #F4A261;">{kategori}</span></h3>
                </div>
                """, unsafe_allow_html=True)

            # Rekomendasi
            st.info(f"💡 **Saran Barista:** {rekomendasi}")

            # --- FEATURE IMPORTANCE CHART ---
            st.markdown("### 📊 Analisis Faktor Penentu")
            
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = model.coef_
            else:
                importances = None

            if importances is not None:
                feature_names = preprocessor.get_feature_names_out()
                feature_names = [f.split("__")[-1] for f in feature_names]

                fi_df = pd.DataFrame({"Faktor": feature_names, "Pengaruh": importances})

                if mode == "akurat":
                    fi_df = fi_df[~fi_df["Faktor"].str.startswith("Processing Method_")]
                    fi_df = fi_df[~fi_df["Faktor"].str.startswith("Variety_")]

                fi_df = fi_df.sort_values("Pengaruh", ascending=True)

                # Plotly dengan Warna Tema Kopi
                fig = px.bar(
                    fi_df, x="Pengaruh", y="Faktor", orientation='h',
                    text="Pengaruh",
                    color_discrete_sequence=['#D4A373'] # Warna Emas/Kopi
                )
                
                # Update Layout agar transparan dan teks terlihat di dark mode
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E6D7C3'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")

else:
    # Tampilan awal sebelum prediksi (Placeholder)
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #888;">
        <h4>👈 Masukkan data kopi di panel sebelah kiri, lalu klik tombol Analisa.</h4>
    </div>
    """, unsafe_allow_html=True)