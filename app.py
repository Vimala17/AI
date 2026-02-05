from http.client import HTTPException
import os
import io
import json
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
# cspell:ignore mobilenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = 'AgriSense AI Assistant' # నేరుగా పేరు ఇచ్చేయ్

groq_api_key = "" # <--- ఇక్కడ నీ ఒరిజినల్ కీ ని పేస్ట్ చెయ్

if groq_api_key == "gsk_your_actual_key_here":
    print("❌ ERROR: బాబు, నువ్వు ఇంకా కీ పేస్ట్ చేయలేదు!")
else:
    print(f"✅ SUCCESS: Groq API Key సెట్ అయింది!")

client = Groq(api_key=groq_api_key)

# --- CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU') 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

# --- 1. SETTING UP PATHS ---
BASE_DIR = Path(r"C:\Users\Acer\OneDrive\PROJECTS\AgriSense AI")
MODELS_DIR = BASE_DIR / "backend" / "models"
FAQ_PATH = MODELS_DIR / "agriculture_faq.csv"
DATASET_PATH = Path(r"C:\Users\Acer\OneDrive\PROJECTS\AgriSense AI\datasets\Crop_recommendation.csv")
CROP_MODEL_PATH = MODELS_DIR / "crop_model.pkl"
DISEASE_MODEL_PATH = MODELS_DIR / "plant_disease_model.h5"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

# --- 2. LOADERS & RESOURCE INITIALIZATION ---
def load_pkl(path):
    if path.exists():
        with open(path, 'rb') as f: return pickle.load(f)
    return None

def load_h5(path):
    if path.exists(): return tf.keras.models.load_model(str(path), compile=False)
    return None

crop_model = load_pkl(CROP_MODEL_PATH)
disease_model = load_h5(DISEASE_MODEL_PATH)


# Disease Labels Load
class_mapping = {}
if CLASS_INDICES_PATH.exists():
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_mapping = json.load(f)
    print("✅ Disease Labels Loaded!")
else:
    class_mapping = { "0": "Apple Scab", "1": "Apple Black Rot", "2": "Cedar Apple Rust", "3": "Healthy" }

faq_df = None
if FAQ_PATH.exists():
    try:
        faq_df = pd.read_csv(FAQ_PATH, encoding='latin-1') 
        print(f"✅ FAQ Knowledge Base Loaded!")
    except Exception as e:
        print(f"❌ FAQ Load Error: {e}")

df_crop = None
if DATASET_PATH.exists():
    try:
        df_crop = pd.read_csv(DATASET_PATH)
        print(f"✅ Crop Dataset Loaded!")
    except Exception as e:
        print(f"❌ Crop Dataset Load Error: {e}")


# --- 4. API ROUTES ---

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "AgriSense AI API Running"}), 200

@app.route('/predict-crop', methods=['POST'])

def predict_crop():
    try:
        d = request.json
        feat = [float(d['nitrogen']), float(d['phosphorus']), float(d['potassium']),
                float(d['temperature']), float(d['humidity']), float(d['ph']), float(d['rainfall'])]
        if crop_model:
            res = crop_model.predict([feat])
            return jsonify({'prediction': str(res[0]), 'status': 'success'})
        elif df_crop is not None:
            X = df_crop.iloc[:, :-1].values
            distances = np.linalg.norm(X - feat, axis=1)
            prediction = df_crop.iloc[np.argmin(distances), -1]
            return jsonify({'prediction': str(prediction), 'status': 'dataset_match'})
        return jsonify({'error': 'No model/data'}), 500
    except Exception as e: return jsonify({'error': str(e)}), 400

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
    
        img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((224, 224))
        img_arr = np.array(img)
        img_arr = preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        preds = disease_model.predict(img_arr, verbose=0)
        pred_idx = str(np.argmax(preds[0]))
        prediction = class_mapping.get(pred_idx, "Unknown Disease")
        confidence = float(np.max(preds[0]))
        
        return jsonify({'prediction': prediction, 'confidence': confidence})
    except Exception as e:
        print(f"❌ Actual Error: {e}") 
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly and professional agricultural expert. "
                        "Respond to the user's query in a naturally, real-world tone. "
                        "Format your response as follows: "
                        "1. Provide the answer in English first. "
                        "2. Leave exactly one blank line gap. "
                        "3. Provide the exact same answer translated into Telugu. "
                        "Do not use markdown like asterisks (**) or bold text. Keep it plain and neat."
                        "Keep the language simple, helpful, and natural, like talking to a farmer friend."
                    )
                },
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        ai_response = chat_completion.choices[0].message.content
        return jsonify({"response": ai_response})

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"response": "Sorry, I'm having trouble connecting right now.\n\nక్షమించండి, ప్రస్తుతం కనెక్ట్ అవ్వడంలో సమస్యగా ఉంది."})
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)