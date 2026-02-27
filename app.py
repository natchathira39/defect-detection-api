import os
import gdown
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

# ── Model config ──────────────────────────────────────────────────────────────
# Replace the file ID below with your own Google Drive file ID
# Drive link format: https://drive.google.com/file/d/YOUR_FILE_ID/view
GDRIVE_FILE_ID = "1C33LGUFb_l4LffZwFvDoWcYR4PtrGBnQ"
MODEL_PATH = "best_model_20260208_073549.h5"

# ── Download model from Google Drive on startup (if not already present) ─────
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )
    print("Model downloaded successfully.")

# ── Load model ────────────────────────────────────────────────────────────────
# Import TensorFlow after download to avoid long startup on first boot
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded and ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
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
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0          # normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0) # shape: (1, 224, 224, 3)

        # Predict
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        # Binary classification logic
        # confidence > 0.5  →  GOOD  (model outputs high value for good disc)
        # confidence < 0.5  →  DEFECTIVE
        if confidence > 0.5:
            result = "GOOD"
        else:
            result = "DEFECTIVE"
            confidence = 1.0 - confidence  # flip so confidence always reflects the result

        return JSONResponse({
            "result": result,
            "confidence": round(confidence * 100, 2),  # return as percentage
            "raw_score": round(float(prediction[0][0]), 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
