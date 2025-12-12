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
    h1 { color: #D4A373 !important; font-family: 'Helvetica', sans-serif; font-weight: 700; }
    h2, h3 { color: #E6D7C3 !important; }
    div.stButton > button:first-child {
        background-color: #D4A373; color: #2C221E; font-weight: bold; border-radius: 10px; border: none; padding: 10px 24px; transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover { background-color: #E9C46A; color: #1E1614; transform: scale(1.02); }
    .result-card { background-color: #2C221E; padding: 20px; border-radius: 15px; border: 1px solid #D4A373; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_h1, col_h2 = st.columns([1, 5])
with col_h1:
    st.image("https://cdn-icons-png.flaticon.com/512/2935/2935413.png", width=80)
with col_h2:
    st.title("AI Coffee Cupping Predictor")
    st.markdown("Prediksi kualitas dan skor kopi Anda menggunakan kecerdasan buatan.")

st.markdown("---")

# ============================
# Default fitur utama
# ============================
DEFAULT_FEATURES = {
    "Altitude": 0.0, "Coffee Age": 0.0, "Moisture Percentage": 0.0,
    "Category One Defects": 0.0, "Category Two Defects": 0.0, "Quakers": 0.0,
    "Uniformity": 0.0, "Clean Cup": 0.0, "Sweetness": 0.0, "Overall": 0.0,
    "Flavor": 0.0, "Aftertaste": 0.0, "Balance": 0.0, "Acidity": 0.0, "Aroma": 0.0,
    "Processing Method": "Natural / Dry", "Variety": "Blend",
}

NUMERIC_RANGES = {
    "Altitude": (0.0, 4000.0), "Coffee Age": (0.0, 50.0), "Moisture Percentage": (0.0, 100.0),
    "Category One Defects": (0.0, 100.0), "Category Two Defects": (0.0, 100.0), "Quakers": (0.0, 100.0),
    "Uniformity": (0.0, 10.0), "Overall": (0.0, 10.0), "Flavor": (0.0, 10.0),
    "Aftertaste": (0.0, 10.0), "Balance": (0.0, 10.0), "Acidity": (0.0, 10.0), "Aroma": (0.0, 10.0),
}

PROCESSING_OPTIONS = ["Natural / Dry", "Pulped natural / honey", "Washed / Wet"]
VARIETY_OPTIONS = ["Gesha", "Caturra", "Typica", "Bourbon", "Catuai", "Catimor", "Ethiopian Heirlooms", "SL34", "Other"]

# ============================
# Inisialisasi session state
# ============================
if "features" not in st.session_state:
    st.session_state.features = DEFAULT_FEATURES.copy()

if st.sidebar.button("🔄 Reset"):
    for key in st.session_state.features.keys():
        st.session_state.features[key] = DEFAULT_FEATURES[key]
    st.sidebar.success("✅ Input telah di-reset ke default")

# ============================
# Sidebar Input
# ============================
input_data = {}
for key, default_value in st.session_state.features.items():
    if isinstance(default_value, (int, float)):
        min_v, max_v = NUMERIC_RANGES.get(key, (0.0, 100000.0))
        value = float(default_value)
        value = max(min(value, max_v), min_v)
        input_data[key] = st.sidebar.number_input(key, value=value, min_value=min_v, max_value=max_v, format="%.2f")
    else:
        if key == "Processing Method": options = PROCESSING_OPTIONS
        elif key == "Variety": options = VARIETY_OPTIONS
        else: options = [default_value]
        try: idx = options.index(default_value)
        except: idx = 0
        input_data[key] = st.sidebar.selectbox(key, options, index=idx)

st.session_state.features = input_data.copy()

# ============================
# Load model & preprocessor
# ============================
try:
    # Kita panggil tanpa mode, karena model_loader sudah diubah
    model, preprocessor = load_model_and_preprocessor()
except Exception as e:
    st.error(f"❌ Gagal load model/preprocessor: {e}")
    st.stop()

# ============================
# PERBAIKAN: Buat input_df di sini
# ============================
try:
    # Panggil fungsi preprocess_single untuk mengubah input user jadi DataFrame
    input_df = preprocess_single(input_data)
except Exception as e:
    st.error(f"Gagal memproses input data: {e}")
    st.stop()

# ============================
# Prediksi
# ============================
if st.button("🔍 Prediksi Sekarang"):
    try:
        # Transform pakai Preprocessor yang diload
        X = preprocessor.transform(input_df)
        
        # Predict
        y_pred = float(model.predict(X)[0])

        st.success(f"### 🎉 Prediksi Cupping Score: **{y_pred:.2f}**")

        kategori = quality_category(y_pred)
        rekomendasi = recommendation_from_category(kategori)

        st.info(f"**Kategori:** {kategori}")
        st.info(f"**Rekomendasi:** {rekomendasi}")

    except Exception as e:
        st.error(f"❌ Error saat prediksi: {e}")

# ============================
# Feature Importance
# ============================
st.markdown("---")
st.header("📊 Feature Importance")

try:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_.flatten()

    if importances is not None:
        try:
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [f.split("__")[-1] for f in feature_names]
        except:
            # Fallback jika get_feature_names_out gagal
            feature_names = list(input_df.columns)

        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance")

        fig = px.bar(
            fi_df, x="Importance", y="Feature",
            title="Feature Importance", orientation="h"
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ Gagal memuat feature importance: {e}")