import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow import keras
from PIL import Image
import tensorflow as tf
import numpy as np
import io

# -----------------------
# Streamlit page config
# -----------------------
st.set_page_config(page_title="Kikoopi Bean Classifier", layout="centered")
st.title("☕ Kikoopi Bean Classifier")
st.write("Upload gambar biji kopi untuk diprediksi kelas dan confidence-nya.")

# -----------------------
# Load model dari HF repo
# -----------------------
@st.cache_resource(show_spinner=True)
def load_model_from_hf():
    try:
        model_path = hf_hub_download(
            repo_id="ikadekranggaa/coffe-bean-classifier",
            filename="best_model.keras"
        )
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal load model dari HF: {e}")
        return None

model = load_model_from_hf()

# -----------------------
# Class labels
# -----------------------
CLASS_NAMES = ["defect", "longberry", "peaberry", "premium"]

# -----------------------
# Image preprocessing
# -----------------------
def preprocess_image(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image = image.resize((224, 224))
        array = tf.keras.preprocessing.image.img_to_array(image)
        array = array / 255.0
        array = np.expand_dims(array, 0)
        return array
    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
        return None

# -----------------------
# Prediction function
# -----------------------
def predict_image(file_bytes):
    x = preprocess_image(file_bytes)
    if x is None:
        return None, None

    try:
        preds = model.predict(x, verbose=0)
        predicted_index = int(tf.argmax(preds[0]))
        confidence = preds[0][predicted_index] * 100
        predicted_class = CLASS_NAMES[predicted_index]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Gagal memprediksi gambar: {e}")
        return None, None

# -----------------------
# Streamlit UI
# -----------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        if model is None:
            st.error("Model belum berhasil di-load. Cek koneksi HF repo.")
        else:
            file_bytes = uploaded_file.getvalue()
            with st.spinner("Sedang memprediksi..."):
                predicted_class, confidence = predict_image(file_bytes)

            if predicted_class is not None:
                st.success(f"Prediction: **{predicted_class}**")
                st.info(f"Confidence: **{confidence:.2f}%**")
            else:
                st.error("Prediksi gagal. Pastikan file gambar valid.")
