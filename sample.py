import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = r'D:\SAFI-Glaucoma-detection\glaucomanew.h5'
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def model_predict(file_path, model):
    processed_img = preprocess_image(file_path)
    preds = model.predict(processed_img)

    # Threshold for considering if it's spam or not (adjust as needed)
    spam_threshold = 0.5

    if preds >= spam_threshold:
        return "glaucoma detected"
    else:
        return "no glaucoma detected"

@app.route('/', methods=['GET'])
def home():
    return render_template('sample.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            result = model_predict(file_path, model)
            return render_template('sample.html', result=result)

    # Handle GET request
    return render_template('sample.html')

if __name__ == '__main__':
    app.run(port=5002, debug=True)
