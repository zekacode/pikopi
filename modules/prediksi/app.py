import streamlit as st
import pandas as pd
import numpy as np
import joblib
from modules.prediction.preprocess import preprocess_batch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px

st.set_page_config(page_title="Coffee Score Prediction", layout="wide")
st.title("Coffee Score Prediction App")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if not uploaded_file:
    st.info("Silakan upload CSV terlebih dahulu.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Data Awal")
st.dataframe(df.head(10))

# -----------------------------
# Preprocessing
# -----------------------------
df_clean = preprocess_batch(df)
st.subheader("Data Setelah Preprocessing")
st.dataframe(df_clean.head(10))

# -----------------------------
# Pilih Mode
# -----------------------------
mode = st.radio("Pilih Mode Prediksi", ["fisik", "akurat"])
if mode == "fisik":
    features = ['Altitude', 'Coffee Age', 'Moisture Percentage',
                'Category One Defects', 'Category Two Defects', 'Quakers']
    model = joblib.load("models/random_forest.pkl")
else:
    features = ['Overall','Flavor','Aftertaste','Balance','Acidity','Aroma','Body']
    features = [f for f in features if f in df_clean.columns]
    model = joblib.load("models/ridge.pkl")

# -----------------------------
# Siapkan Data untuk Prediksi
# -----------------------------
numeric_features = features
categorical_features = [c for c in ['Processing Method','Variety'] if c in df_clean.columns]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])

X_processed = preprocessor.fit_transform(df_clean[features + categorical_features])

# -----------------------------
# Prediksi
# -----------------------------
df_clean['Predicted_Score'] = model.predict(X_processed)

st.subheader("Hasil Prediksi")
columns_to_show = ['Predicted_Score'] + categorical_features
st.dataframe(df_clean[columns_to_show].head(10))

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance")
try:
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        st.info("Feature importance tidak tersedia.")
        importance = None

    if importance is not None:
        feat_names = preprocessor.get_feature_names_out()
        feat_df = pd.DataFrame({
            'Feature': [f.replace('cat__','').replace('num__','') for f in feat_names],
            'Importance': importance
        }).sort_values(by='Importance', ascending=True)
        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title=f"Feature Importance ({mode})")
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Gagal menampilkan feature importance: {e}")

# -----------------------------
# Download Hasil
# -----------------------------
csv = df_clean.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Hasil Prediksi CSV",
    data=csv,
    file_name="prediksi_coffee.csv",
    mime="text/csv"
)
