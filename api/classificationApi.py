from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "best_model.keras")

model = load_model(MODEL_DIR)

CLASS_NAMES = ["defect", "longberry", "peaberry", "premium"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, 0)
    return image

@app.get("/")
def read_root():
    return {"message": "Welcome to the Kikoopi Bean Classifier API"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        x = preprocess_image(content)
        preds = model.predict(x, verbose=0)
        predicted_index = int(tf.argmax(preds[0]))
        confidence_value = preds[0][predicted_index].item() * 100
        predicted_class = CLASS_NAMES[predicted_index]

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": f"{confidence_value:.2f}%"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("classificationApi:app", host="0.0.0.0", port=8000, reload=True)
