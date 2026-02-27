import os
import gdown
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

GDRIVE_FILE_ID = "1C33LGUFb_l4LffZwFvDoWcYR4PtrGBnQ"
MODEL_PATH = "best_model_20260208_073549.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        id=GDRIVE_FILE_ID,
        output=MODEL_PATH,
        quiet=False,
        fuzzy=True
    )
    print("Model downloaded successfully.")

import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded and ready.")

app = FastAPI(title="Disc Brake Defect Detection API")

@app.get("/")
def root():
    return {"status": "Disc Brake Defect Detection API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            result = "GOOD"
        else:
            result = "DEFECTIVE"
            confidence = 1.0 - confidence

        return JSONResponse({
            "result": result,
            "confidence": round(confidence * 100, 2),
            "raw_score": round(float(prediction[0][0]), 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
