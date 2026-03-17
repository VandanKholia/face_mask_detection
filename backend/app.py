import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_PATH = '../face_mask_model.h5'
model = None

# Load model before first request
def load_keras_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

load_keras_model()

def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))
    img = img.convert('RGB')
    img_array = np.array(img)
    img_array = img_array / 255.0  # Scale pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            processed_image = preprocess_image(file)
            prediction = model.predict(processed_image)
            
          
            
            label_index = np.argmax(prediction)
            
            label = "With Mask" if label_index == 1 else "Without Mask"
            confidence = float(np.max(prediction))

            return jsonify({
                'label': label,
                'confidence': confidence,
                'label_index': int(label_index)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
