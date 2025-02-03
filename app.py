from flask import Flask, render_template, request
import os
import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'

# Periksa apakah folder sudah ada sebelum mencoba membuatnya
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Registrasikan Custom Loss Function agar bisa dikenali saat Load Model
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Load Model CNN (Pastikan model sudah dilatih dan disimpan)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Direktori utama proyek
MODEL_PATH = os.path.join(BASE_DIR, "model_age_cnn.h5")  # Jalur absolut

# Pastikan model ada sebelum diload
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"custom_mse": custom_mse, "MeanSquaredError": tf.keras.losses.MeanSquaredError})
else:
    raise FileNotFoundError(f"❌ Model tidak ditemukan di {MODEL_PATH}. Pastikan 'model_age_cnn.h5' ada.")

# Fungsi Preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = np.expand_dims(img, axis=-1)  # Tambah channel
    img = np.expand_dims(img, axis=0)   # Tambah batch
    img = img / 255.0  # Normalisasi
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html', title="Tentang Aplikasi", description="""
        Deep Learning Age App adalah aplikasi berbasis web yang menggunakan teknologi Convolutional Neural Network (CNN) 
        untuk melakukan prediksi umur seseorang berdasarkan gambar wajah mereka. Aplikasi ini membantu dalam identifikasi usia 
        berdasarkan pola wajah dan dapat diterapkan dalam berbagai keperluan analisis citra.

        Fitur utama dari aplikasi ini:
        1. Upload Gambar: Pengguna dapat mengunggah gambar wajah mereka.
        2. Prediksi Umur: Model CNN akan memproses gambar dan memprediksi umur.
        3. Tampilan Responsif: Antarmuka pengguna dibuat agar mudah digunakan di berbagai perangkat.

        Teknologi yang digunakan dalam aplikasi ini meliputi Flask sebagai backend, TensorFlow untuk deep learning, 
        dan OpenCV untuk pemrosesan gambar.
    """)

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess dan Prediksi
        img = preprocess_image(filepath)
        prediction = model.predict(img)
        predicted_age = int(prediction[0][0])
        
        return render_template('result.html', age=predicted_age, image=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
