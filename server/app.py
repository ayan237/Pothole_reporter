from flask import Flask, request, jsonify, session
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import secrets

from auth import init_db, add_user, get_user, verify_user

# --- ML Model Setup ---
MODEL_PATH = 'pothole_severity_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
class_map = {0: "Minor", 1: "Moderate", 2: "Severe"}

def predict_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224)).convert('RGB')
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    return {
        "severity_class": class_idx,
        "severity_label": class_map[class_idx],
        "confidence": confidence
    }

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)


# --- Routes ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    if add_user(username, password):
        return jsonify({'message': 'User registered successfully'})
    else:
        return jsonify({'error': 'Username already exists'}), 409

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = verify_user(username, password)
    if user:
        session['user_id'] = user[0]
        session['username'] = user[1]
        return jsonify({'message': 'Login successful'})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'})

@app.route('/predict', methods=['POST'])
def predict():
    # Optionally restrict prediction to logged-in users:
    # if 'user_id' not in session:
    #     return jsonify({'error': 'Authentication required'}), 401

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    result = predict_image_from_bytes(image_bytes)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
