import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

model_path = r"C:\Users\preml\Desktop\Flask-Plant_API\model.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
interpreter = tf.lite.Interpreter(model_path=model_path)

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app access

# Load the TensorFlow Lite model
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\preml\Desktop\Flask-Plant_API\model.tflite")  
interpreter.allocate_tensors()
print("TFLite model loaded successfully!")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    
    # Read image and preprocess
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))  # Resize to model input size
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)

    if predicted_index >= len(class_names):
        return jsonify({"error": f"Invalid prediction index: {predicted_index}"}), 500

    predicted_class = class_names[predicted_index]
    
    return jsonify({"predicted_species": predicted_class})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment
    app.run(host="0.0.0.0", port=port)

