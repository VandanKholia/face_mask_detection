import os
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'face_mask_model.keras')


model = load_model(MODEL_PATH, compile=False)

def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        processed_image = preprocess_image(file)
        prediction = model.predict(processed_image)

        label_index = np.argmax(prediction)
        label = "With Mask" if label_index == 1 else "Without Mask"
        confidence = float(np.max(prediction))

        return jsonify({
            'label': label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)