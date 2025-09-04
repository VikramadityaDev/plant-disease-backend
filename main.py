# Plant Disease Detection API
# ===================================
# Author   : Vikramaditya
# Course   : Minor Project
# Version  : 1.0
# Date     : September 2025
# Framework: FastAPI + TensorFlow
# ===================================

import os
import gdown
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import json

app = FastAPI()

# -----------------------------
# Enable CORS for application
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Download model if not exists
# ------------------------------
MODEL_PATH = "plant_disease_prediction_model.h5"
FILE_ID = "1kp5wfpdZ787eflIpb9CAwF1m0G2ZFlTS"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

# ----------------------
# Load model from here
# ----------------------
model = tf.keras.models.load_model(MODEL_PATH)

# --------------------
# Load class indices
# --------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = {int(k): v for k, v in class_indices.items()}

# -----------------
# Treatment dict.
# -----------------
TREATMENTS = {
    "Apple___Apple_scab": "Use fungicides like Captan or Mancozeb. Remove and destroy fallen leaves.",
    "Apple___Black_rot": "Prune infected branches. Apply fungicides such as Mancozeb or Ziram.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees. Use fungicides (e.g., Myclobutanil).",
    "Apple___healthy": "The plant is healthy. Maintain proper pruning and fertilization.",
    "Blueberry___healthy": "The plant is healthy. Ensure acidic soil and proper irrigation.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur fungicide. Ensure good airflow by pruning.",
    "Cherry_(including_sour)___healthy": "Healthy! Maintain good pruning and irrigation practices.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, use resistant varieties, apply fungicides if severe.",
    "Corn_(maize)___Common_rust_": "Use resistant hybrids. Apply fungicides only if severe infection.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids, rotate crops, apply fungicides if necessary.",
    "Corn_(maize)___healthy": "Healthy! Maintain spacing and balanced fertilization.",
    "Grape___Black_rot": "Remove mummified fruit and infected leaves. Apply fungicides like Mancozeb.",
    "Grape___Esca_(Black_Measles)": "No complete cure. Prune infected vines and avoid drought stress.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides and remove affected leaves.",
    "Grape___healthy": "The plant is healthy. Maintain canopy airflow.",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure. Remove infected trees and control psyllid vectors with insecticides.",
    "Peach___Bacterial_spot": "Apply copper-based sprays. Use resistant cultivars.",
    "Peach___healthy": "Healthy! Maintain fertilization and irrigation schedule.",
    "Pepper,_bell___Bacterial_spot": "Remove infected plants. Apply copper fungicides.",
    "Pepper,_bell___healthy": "Healthy! Monitor regularly and maintain proper watering.",
    "Potato___Early_blight": "Remove infected leaves. Apply fungicides like Chlorothalonil or Mancozeb.",
    "Potato___Late_blight": "Destroy infected plants. Apply fungicides (e.g., Mancozeb, Metalaxyl). Avoid overhead irrigation.",
    "Potato___healthy": "The plant is healthy. Rotate crops and monitor leaf health.",
    "Raspberry___healthy": "The plant is healthy. Maintain pruning and good spacing.",
    "Soybean___healthy": "The plant is healthy. Rotate crops and monitor aphids.",
    "Squash___Powdery_mildew": "Apply sulfur or neem oil. Increase airflow between plants.",
    "Strawberry___Leaf_scorch": "Remove affected leaves. Ensure proper watering without overhead irrigation.",
    "Strawberry___healthy": "The plant is healthy. Mulch soil to retain moisture.",
    "Tomato___Bacterial_spot": "Remove infected plants. Apply copper-based fungicides.",
    "Tomato___Early_blight": "Remove infected leaves. Apply fungicides (Chlorothalonil, Mancozeb). Rotate crops.",
    "Tomato___Late_blight": "Destroy infected plants. Apply fungicides (Copper, Mancozeb). Avoid leaf wetness.",
    "Tomato___Leaf_Mold": "Improve ventilation. Apply fungicides like Chlorothalonil.",
    "Tomato___Septoria_leaf_spot": "Prune affected leaves. Use fungicides like Mancozeb or Chlorothalonil.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray with miticides or neem oil. Increase humidity.",
    "Tomato___Target_Spot": "Use fungicides like Mancozeb. Rotate crops to reduce recurrence.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants. Control whiteflies with insecticides.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants. Disinfect tools. Use resistant cultivars.",
    "Tomato___healthy": "The plant is healthy. Ensure consistent watering and staking."
}

# ---------------------
# Preprocess function
# ---------------------
def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ------------------
# Predict endpoint
# ------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    processed = preprocess(image)
    predictions = model.predict(processed)

    idx = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    disease = class_names[idx]

    treatment = TREATMENTS.get(disease, "No treatment information available.")

    return {
        "disease": disease,
        "confidence": confidence,
        "treatment": treatment
    }
