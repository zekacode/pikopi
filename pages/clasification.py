import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow import keras
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import time

# --- 1. KONFIGURASI HALAMAN & CSS ---
st.set_page_config(page_title="Kikoopi Bean Classifier", layout="wide", page_icon="☕")

# Custom CSS (Tema PIKOPI)
st.markdown("""
<style>
    /* Judul H1 */
    h1 {
        color: #D4A373 !important;
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
    }
    
    /* Tombol Utama */
    div.stButton > button:first-child {
        background-color: #D4A373;
        color: #2C221E;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #E9C46A;
        color: #1E1614;
        transform: scale(1.02);
    }

    /* Card Hasil */
    .result-card {
        background-color: #2C221E;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #D4A373;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Teks Hasil */
    .result-title { font-size: 1.2em; color: #E6D7C3; margin-bottom: 5px; }
    .result-value { font-size: 2.5em; font-weight: bold; color: #D4A373; margin: 0; }
    .result-conf { font-size: 1em; color: #F4A261; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_h1, col_h2 = st.columns([1, 6])
with col_h1:
    st.image("https://cdn-icons-png.flaticon.com/512/4856/4856703.png", width=90) # Icon Scanner
with col_h2:
    st.title("AI Bean Classifier")
    st.markdown("Deteksi jenis dan kualitas biji kopi Anda menggunakan *Computer Vision*.")

st.markdown("---")

# --- 2. LOAD MODEL (CACHED) ---
@st.cache_resource(show_spinner=False)
def load_model_from_hf():
    try:
        with st.spinner("Sedang memanaskan mesin AI..."):
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

# --- 3. LOGIKA PREDIKSI ---
CLASS_NAMES = ["Defect ❌", "Longberry 🍇", "Peaberry 🍒", "Premium 💎"]

# Deskripsi untuk setiap kelas (Barista Notes)
CLASS_DESC = {
    "Defect ❌": "Biji kopi cacat (pecah, berlubang, atau jamuran). Sebaiknya disortir agar tidak merusak rasa.",
    "Longberry 🍇": "Biji kopi yang bentuknya lonjong memanjang. Biasanya punya karakter rasa yang unik dan floral.",
    "Peaberry 🍒": "Biji kopi tunggal (jantan). Bentuknya bulat utuh, sering dianggap memiliki rasa yang lebih intens.",
    "Premium 💎": "Biji kopi kualitas terbaik dengan bentuk sempurna. Siap untuk disangrai (roasting)!"
}

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

def predict_image(file_bytes):
    x = preprocess_image(file_bytes)
    if x is None: return None, None
    
    try:
        preds = model.predict(x, verbose=0)
        predicted_index = int(tf.argmax(preds[0]))
        confidence = preds[0][predicted_index] * 100
        predicted_class = CLASS_NAMES[predicted_index]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Gagal memprediksi gambar: {e}")
        return None, None

# --- 4. UI UTAMA ---

# Sidebar Info
with st.sidebar:
    st.header("📸 Panduan Foto")
    st.info("""
    1. Pastikan biji kopi terlihat jelas.
    2. Gunakan pencahayaan yang cukup.
    3. Foto satu biji atau sekumpulan biji sejenis.
    """)
    st.markdown("---")
    st.caption("Powered by TensorFlow & Hugging Face")

# Layout 2 Kolom
col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.subheader("1. Upload Foto")
    uploaded_file = st.file_uploader("Pilih gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Preview Gambar", use_container_width=True)

with col_result:
    st.subheader("2. Hasil Analisa")
    
    if uploaded_file:
        # Tombol Predict
        if st.button("🔍 Identifikasi Biji Kopi"):
            if model is None:
                st.error("Model belum siap. Cek koneksi internet.")
            else:
                file_bytes = uploaded_file.getvalue()
                
                # Progress Bar Effect
                progress_text = "Sedang mengamati tekstur biji..."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                my_bar.empty()

                # Prediksi
                predicted_class, confidence = predict_image(file_bytes)

                if predicted_class is not None:
                    # Tampilkan Hasil dalam Card
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">Jenis Biji Terdeteksi</div>
                        <div class="result-value">{predicted_class}</div>
                        <div class="result-conf">Tingkat Keyakinan: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress Bar Confidence
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write("Akurasi Model:")
                    st.progress(int(confidence), text=f"{confidence:.2f}%")
                    
                    # Barista Note
                    st.success(f"💡 **Catatan Barista:**\n\n{CLASS_DESC.get(predicted_class, '')}")
                    
                else:
                    st.error("Gagal mengenali gambar.")
    else:
        # Placeholder jika belum upload
        st.info("👈 Silakan upload foto biji kopi di sebelah kiri untuk melihat hasil analisa di sini.")