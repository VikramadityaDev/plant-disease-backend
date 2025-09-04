from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np
import json

app = FastAPI()

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Load class mapping
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}  # reverse mapping

# prepocessing the image
def preprocess(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# endpoint 
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    processed = preprocess(image)
    predictions = model.predict(processed)
    idx = int(np.argmax(predictions))
    return {
        "disease": class_names[idx],
        "confidence": float(np.max(predictions))
    }

