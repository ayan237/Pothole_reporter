from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model and class map
MODEL_PATH = 'pothole_severity_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
class_map = {0: "Minor", 1: "Moderate", 2: "Severe"}

def predict_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224,224)).convert('RGB')
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    class_idx = np.argmax(preds)
    confidence = float(preds[class_idx])
    return {
        "severity_class": int(class_idx),
        "severity_label": class_map[class_idx],
        "confidence": confidence
    }

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    result = predict_image_from_bytes(image_bytes)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)