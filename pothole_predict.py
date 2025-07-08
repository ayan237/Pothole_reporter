import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
MODEL_PATH = 'pothole_severity_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Class mapping
class_map = {0: "Minor", 1: "Moderate", 2: "Severe"}

def predict_image(image_path):
    img = Image.open(image_path).resize((224,224)).convert('RGB')
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    class_idx = np.argmax(preds)
    confidence = float(preds[class_idx])
    result = {
        "severity_class": int(class_idx),
        "severity_label": class_map[class_idx],
        "confidence": confidence
    }
    return json.dumps(result, indent=2)

# Example usage
if __name__ == "__main__":
    image_path = r"E:\HackOrbit\classified_dataset\Severe\256.jpg"  # Replace with your image
    print(predict_image(image_path))