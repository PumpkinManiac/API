import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, force CPU usage

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app access

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model("medicinal_plant_classifier.h5")
print("Model loaded successfully!")

# Define class names (Update based on dataset)
class_names = [
    "Neem", "Tulsi", "Aloe Vera", "Ashwagandha", "Brahmi",
    "Class6", "Class7", "Class8", "Class9", "Class10",
    "Class11", "Class12", "Class13", "Class14", "Class15",
    "Class16", "Class17", "Class18", "Class19", "Class20",
    "Class21", "Class22", "Class23", "Class24", "Class25",
    "Class26", "Class27", "Class28", "Class29", "Class30",
    "Class31", "Class32", "Class33", "Class34", "Class35",
    "Class36", "Class37", "Class38", "Class39", "Class40"
] 

@app.route('/')
def home():
    return jsonify({"message": "Plant Species Prediction API is Running!"})


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    predictions = model.predict(img)
    predicted_index = np.argmax(predictions)

    if predicted_index >= len(class_names):
        return jsonify({"error": f"Invalid prediction index: {predicted_index}"}), 500

    predicted_class = class_names[predicted_index]
    
    return jsonify({"predicted_species": predicted_class})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
