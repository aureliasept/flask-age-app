from flask import Flask, render_template, request # type: ignore
import os
import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from werkzeug.utils import secure_filename # type: ignore
from waitress import serve  # type: ignore # Pastikan waitress sudah terinstal

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'

# Pastikan folder untuk upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi Custom Loss agar bisa dikenali saat Load Model
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# 🔹 **Memuat Model CNN**
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Direktori utama proyek
MODEL_PATH = os.path.join(BASE_DIR, "model_age_cnn.h5")  # Jalur absolut

# Pastikan model ada sebelum diload
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"custom_mse": custom_mse, "MeanSquaredError": tf.keras.losses.MeanSquaredError}
    )

    # 🔹 **Kompilasi ulang model agar metrik tidak kosong**
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    print("✅ Model berhasil dimuat dan dikompilasi ulang.")
else:
    raise FileNotFoundError(f"❌ Model tidak ditemukan di {MODEL_PATH}. Pastikan 'model_age_cnn.h5' ada.")

# 🔹 **Fungsi Preprocessing Gambar**
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
    Aplikasi Identifikasi Umur adalah aplikasi berbasis web yang memanfaatkan teknologi Convolutional Neural Network (CNN) untuk melakukan prediksi umur seseorang berdasarkan gambar wajah mereka. 

    Fitur utama dari aplikasi ini:
    1. Upload Gambar: Pengguna dapat mengunggah gambar wajah mereka.
    2. Prediksi Umur: Model CNN yang telah dilatih akan menganalisis gambar dan memberikan estimasi umur.
    3. Tampilan Responsif: Hasil prediksi ditampilkan dalam antarmuka yang mudah digunakan.

    Aplikasi ini menggunakan teknologi seperti Flask sebagai backend, TensorFlow untuk deep learning, dan OpenCV untuk pemrosesan gambar. Dengan model yang terus disempurnakan, aplikasi ini dapat menjadi contoh bagaimana AI dapat digunakan dalam bidang identifikasi wajah dan analisis umur.
    """)

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # **🔹 Preprocess dan Prediksi**
        img = preprocess_image(filepath)
        prediction = model.predict(img)
        predicted_age = int(prediction[0][0])
        
        return render_template('result.html', age=predicted_age, image=filepath)
    
    return render_template('index.html')

if __name__ == '__main__':
    print("🚀 Server berjalan di http://127.0.0.1:8080")
    serve(app, host="0.0.0.0", port=8080)
