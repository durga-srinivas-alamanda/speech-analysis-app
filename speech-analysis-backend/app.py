from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import joblib
import warnings
import os
import logging
warnings.simplefilter("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Ensure uploads directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Speech-to-Text Model
try:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model_sst = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    logging.info("Speech-to-Text model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading SST model: {e}")
    raise e

# Load Speech Emotion Recognition Model
try:
    model_ser = joblib.load("ser_model.pkl")  # Ensure model path is correct
    logging.info("Speech Emotion Recognition model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading SER model: {e}")
    raise e

# Define emotion labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

# Function to extract features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

@app.route("/process_audio", methods=["POST"])
def process_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)
    logging.info(f"Received and saved audio file: {audio_file.filename}")
    
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Speech-to-Text Conversion
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model_sst(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        # Speech Emotion Recognition
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=0)
        emotion_pred = model_ser.predict(features)[0]
        emotion_index = np.argmax(emotion_pred)
        emotion_label = EMOTION_LABELS[emotion_index]
        
        response = {
            "filename": audio_file.filename,
            "transcription": transcription,
            "emotion": emotion_label,
            "audio_url": f"/uploads/{audio_file.filename}"
        }
        logging.info(f"Processed audio: {response}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error processing audio file: {e}")
        return jsonify({"error": "Failed to process audio"}), 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Production-ready deployment
