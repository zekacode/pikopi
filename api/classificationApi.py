from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download
import io
import os

app = FastAPI()

# ==========================
# CORS
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["defect", "longberry", "peaberry", "premium"]

model = None  # lazy load


# ==========================
# LOAD MODEL SAAT STARTUP
# ==========================
@app.on_event("startup")
def load_model():
    global model
    model_path = hf_hub_download(
        repo_id="ikadekranggaa/coffe-bean-classifier",
        filename="best_model.keras",
    )
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")


# ==========================
# HELPER
# ==========================
def preprocess_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, 0)
    return image


# ==========================
# ROUTES
# ==========================
@app.get("/")
def read_root():
    return {"message": "Welcome to the Kikoopi Bean Classifier API!"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        x = preprocess_image(content)
        preds = model.predict(x, verbose=0)
        predicted_index = int(tf.argmax(preds[0]))
        confidence_value = float(preds[0][predicted_index] * 100)

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": f"{confidence_value:.2f}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# UNTUK LOCAL ONLY
# ==========================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))  # Railway uses $PORT
    uvicorn.run("api.classificationApi:app", host="0.0.0.0", port=port, reload=True)
