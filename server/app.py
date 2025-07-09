from flask import Flask, request, jsonify, session, send_from_directory
from flask import render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
import os
from datetime import datetime
import io
import secrets

from auth import init_db, init_reports_table, add_user, get_user, verify_user

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

# --- Auth Routes ---
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
    username = request.form.get('username')
    password = request.form.get('password')
    user = verify_user(username, password)
    if user:
        session['user_id'] = user[0]
        session['username'] = user[1]
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html', error="Invalid credentials")

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login_form'))

# --- Root + Dashboard ---
@app.route('/')
def root():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_form'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        return render_template('home.html', logged_in=True)
    return redirect(url_for('login_form'))

# --- Report Routes ---
@app.route('/report-form')
def report_form():
    if 'user_id' not in session:
        return redirect(url_for('login_form'))
    return render_template('report.html')

@app.route('/report', methods=['POST'])
def report_pothole():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401

    user_id = session['user_id']
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    severity_result = predict_image_from_bytes(image_bytes)

    # Save image
    image_folder = 'report_images'
    os.makedirs(image_folder, exist_ok=True)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"user{user_id}_{timestamp_str}.jpg"
    filepath = os.path.join(image_folder, filename)
    with open(filepath, 'wb') as f:
        f.write(image_bytes)

    # Save to DB
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO reports (user_id, severity_label, confidence, latitude, longitude, timestamp, image_filename)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
    ''', (user_id, severity_result["severity_label"], severity_result["confidence"], latitude, longitude, filename))
    conn.commit()
    report_id = c.lastrowid
    conn.close()

    return jsonify({
        "message": "Report submitted successfully",
        "severity": severity_result,
        "image_filename": filename,
        "report_id": report_id
    })

@app.route('/reports', methods=['GET'])
def get_reports():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, user_id, latitude, longitude, severity_label, confidence, timestamp, image_filename FROM reports')
    rows = c.fetchall()
    conn.close()

    reports = [{
        'id': r[0], 'user_id': r[1], 'latitude': r[2], 'longitude': r[3],
        'severity_label': r[4], 'confidence': r[5], 'timestamp': r[6],
        'image_url': f"/images/{r[7]}" if r[7] else None
    } for r in rows]

    return jsonify({'reports': reports})

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'report_images'), filename)

# --- Status + Tracking ---
@app.route('/status')
def status_form():
    if 'user_id' not in session:
        return redirect(url_for('login_form'))
    return render_template('status.html')

@app.route('/report-status/<int:report_id>')
def report_status(report_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, status, severity_label, timestamp, image_filename FROM reports WHERE id=?', (report_id,))
    report = c.fetchone()
    conn.close()

    if not report:
        return jsonify({"error": "Report not found"}), 404

    return jsonify({
        "id": report[0],
        "status": report[1],
        "severity_label": report[2],
        "timestamp": report[3],
        "image_filename": f"report_images/{report[4]}"
    })

@app.route('/status/<int:report_id>', methods=['GET'])
def track_status(report_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return jsonify({
            "id": row[0],
            "user_id": row[1],
            "latitude": row[2],
            "longitude": row[3],
            "severity_class": row[4],
            "severity_label": row[5],
            "confidence": row[6],
            "timestamp": row[7],
            "image_filename": row[8],
            "status": row[9],
            "upvotes": row[10]
        })
    return jsonify({"error": "Report not found"}), 404

# --- My Reports ---
@app.route('/my-reports')
def my_reports():
    if 'user_id' not in session:
        return redirect(url_for('login_form'))

    user_id = session['user_id']
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, latitude, longitude, severity_label, confidence, timestamp, image_filename, status FROM reports WHERE user_id = ?', (user_id,))
    rows = c.fetchall()
    conn.close()

    reports = [{
        'id': r[0], 'latitude': r[1], 'longitude': r[2],
        'severity_label': r[3], 'confidence': round(r[4], 2),
        'timestamp': r[5], 'image_filename': r[6], 'status': r[7]
    } for r in rows]

    return render_template('my_reports.html', reports=reports)

# --- Form Pages ---
@app.route('/login-form')
def login_form():
    return render_template('login.html')

@app.route('/register-form')
def register_form():
    return render_template('register.html')

# ✅ Init DB + Run Flask Server
if __name__ == '__main__':
    init_db()
    init_reports_table()
    app.run(debug=True)
