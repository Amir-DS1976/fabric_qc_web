from flask import Flask, render_template, request
import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from werkzeug.utils import secure_filename

# --- Configuration ---
IMAGE_SIZE = (128, 128)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
LABEL_MAP = {0: 'CLEAN', 1: 'DEFECTED'}
MODEL_PATH = 'fabric_qc_model.pkl'

# --- Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = joblib.load(MODEL_PATH)

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    # If you trained with raw pixels, use this:
    features = resized.flatten()
    # If you trained with HOG, use this instead:
    # features = hog(resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return features

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file part is present
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process image and predict
            features = preprocess_image(filepath)
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0][prediction]
            label = LABEL_MAP[prediction]

            return render_template('index.html',
                                   filename=filename,
                                   label=label,
                                   probability=f"{probability:.2f}")

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
