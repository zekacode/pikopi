import streamlit as st
import requests

API_URL = "https://pikopi-zpnc533gx7lf9fzd7jiqu9.streamlit.app//predict"

st.title("☕ Kikoopi Bean Classifier")
st.write("Upload gambar biji kopi untuk diprediksi")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")

    if st.button("Predict"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        try:
            res = requests.post(API_URL, files=files)

            if res.status_code == 200:
                data = res.json()
                st.success(f"Prediction: **{data['predicted_class']}**")
                st.info(f"Confidence: **{data['confidence']}**")
            else:
                st.error(f"Prediction failed! Status Code: {res.status_code}")
                st.json(res.text)

        except Exception as e:
            st.error(f"Error: {str(e)}")
