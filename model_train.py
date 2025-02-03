import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
import numpy as np # type: ignore

# Dataset Dummy (Gantilah dengan dataset asli jika tersedia)
X_train = np.random.rand(1000, 48, 48, 1)  # 1000 gambar grayscale 48x48
y_train = np.random.randint(0, 90, 1000)   # Umur acak dari 0 hingga 90 tahun

X_val = np.random.rand(200, 48, 48, 1)     # Data validasi
y_val = np.random.randint(0, 90, 200)

# Gunakan loss function eksplisit
loss_function = tf.keras.losses.MeanSquaredError()

# Definisi Model CNN untuk Prediksi Umur
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')  # Output umur
])

# Kompilasi Model dengan Loss Function yang Benar
model.compile(optimizer='adam', loss=loss_function, metrics=['mae'])

# Latih Model (Gunakan dataset asli jika tersedia)
print("🔄 Melatih model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Simpan Model ke dalam file .h5
model.save('model_age_cnn.h5')
print("✅ Model CNN berhasil disimpan sebagai 'model_age_cnn.h5'")
