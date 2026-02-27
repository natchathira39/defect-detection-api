import os
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

GDRIVE_FILE_ID = "1C33LGUFb_l4LffZwFvDoWcYR4PtrGBnQ"
MODEL_PATH = "best_model_20260208_073549.h5"

def download_from_gdrive(file_id, output_path):
    print("Downloading model from Google Drive...")
    
    session = requests.Session()
    
    # Step 1 — initial request to get confirmation token
    URL = "https://drive.google.com/uc?export=download"
    response = session.get(URL, params={"id": file_id}, stream=True)
    
    # Step 2 — handle large file confirmation page
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    
    # Also check for new-style confirmation in response content
    if token is None:
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                if "confirm=" in decoded:
                    import re
                    match = re.search(r'confirm=([0-9A-Za-z_]+)', decoded)
                    if match:
                        token = match.group(1)
                        break

    if token:
        params = {"id": file_id, "confirm": token, "export": "download"}
    else:
        params = {"id": file_id, "export": "download", "confirm": "t"}

    # Step 3 — actual download
    response = session.get(URL, params=params, stream=True)
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    
    print("Model downloaded successfully.")

if not os.path.exists(MODEL_PATH):
    download_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)

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
