# app.py
import streamlit as st
import pandas as pd
from modules.prediction.preprocess_single import preprocess_single
from modules.prediction.model_loader import load_model_and_preprocessor
from modules.prediction.recommend import quality_category, recommendation_from_category
import plotly.express as px

st.set_page_config(page_title="Prediksi Cupping Score", layout="wide")
st.title("☕ Prediksi Cupping Score Kopi")
st.write("Gunakan model machine learning untuk memprediksi cupping score berdasarkan data kopi.")

# ============================
# Sidebar: Pilih Mode & Reset
# ============================
mode = st.sidebar.selectbox("Pilih Mode Prediksi:", ["Fisik", "Akurat"]).lower()
reset_btn = st.sidebar.button("🔄 Reset Input")

# ============================
# Default fitur untuk masing-masing mode
# ============================
def get_default_features(mode):
    if mode == "fisik":
        return {
            "Altitude": 1200,
            "Coffee Age": 10,
            "Moisture Percentage": 11.0,
            "Category One Defects": 1,
            "Category Two Defects": 2,
            "Quakers": 1,
            "Processing Method": "Natural / Dry",
            "Variety": "Blend",
        }
    else:
        return {
            "Uniformity": 10.0,
            "Clean Cup": 10.0,
            "Sweetness": 10.0,
            "Overall": 8.0,
            "Flavor": 8.0,
            "Aftertaste": 8.0,
            "Balance": 8.0,
            "Acidity": 8.0,
            "Aroma": 8.0,
            "Body": 8.0,
            "Processing Method": "Natural / Dry",
            "Variety": "Blend",
        }

# State Streamlit untuk reset
if "features" not in st.session_state or reset_btn:
    st.session_state.features = get_default_features(mode)

# ============================
# Sidebar input
# ============================
input_data = {}
for key, val in st.session_state.features.items():
    if isinstance(val, float):
        input_data[key] = st.sidebar.number_input(key, value=val)
    elif isinstance(val, int):
        input_data[key] = st.sidebar.number_input(key, value=val, step=1)
    else:
        if key == "Processing Method":
            input_data[key] = st.sidebar.selectbox(
                key,
                ["Natural / Dry", "Pulped natural / honey", "Washed / Wet"],
                index=["Natural / Dry", "Pulped natural / honey", "Washed / Wet"].index(val)
            )
        else:
            input_data[key] = st.sidebar.selectbox(
                key,
                ["Blend", "Catuai", "Other", "Red Bourbon,Caturra", "unknown"],
                index=["Blend", "Catuai", "Other", "Red Bourbon,Caturra", "unknown"].index(val)
            )

# ============================
# Load Model & Preprocessor
# ============================
model, preprocessor = load_model_and_preprocessor(mode)
if model is None or preprocessor is None:
    st.error("❌ Model atau preprocessor tidak ditemukan di folder 'models/'.")
    st.stop()

# ============================
# Preprocess input
# ============================
input_df = preprocess_single(input_data, mode=mode)

# ============================
# Prediksi
# ============================
if st.button("🔍 Prediksi Sekarang"):
    try:
        X = preprocessor.transform(input_df)
        y_pred = model.predict(X)[0]
        st.success(f"### 🎉 Prediksi Cupping Score: **{y_pred:.2f}**")

        kategori = quality_category(y_pred)
        rekomendasi = recommendation_from_category(kategori)
        st.info(f"**Kategori Kualitas:** {kategori}")
        st.info(f"**Rekomendasi:** {rekomendasi}")

    except Exception as e:
        st.error(f"❌ Terjadi error saat prediksi: {e}")

# ============================
# Feature Importance
# ============================
st.markdown("---")
st.header("📊 Feature Importance")

try:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_
    else:
        st.warning("Model tidak memiliki feature importance.")
        importances = None

    if importances is not None:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [f.split("__")[-1] for f in feature_names]  # hapus num__ / cat__

        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

        # Untuk mode akurat: hapus semua kolom kategorikal berdasarkan prefix
        if mode == "akurat":
            fi_df = fi_df[~fi_df["Feature"].str.startswith("Processing Method_")]
            fi_df = fi_df[~fi_df["Feature"].str.startswith("Variety_")]

        fi_df = fi_df.sort_values("Importance", ascending=True)

        # Bar chart pakai Plotly
        import plotly.express as px
        fig = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Feature Importance",
            text="Importance"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Gagal memuat feature importance: {e}")
