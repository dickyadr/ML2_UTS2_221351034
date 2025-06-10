import streamlit as st
import tensorflow  as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="avocado.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Kualitas avocado")
st.write("Masukkan informasi Avocado.")

# Form input pengguna
A = st.number_input("Persentase kemurnian warna (0-100%) (A)", min_value=0.0, max_value=100.0, value=10.0)
B = st.number_input("Persentase intensitas cahaya (0-100%) (B)", min_value=0.0, max_value=100.0, value=10.0)
C = st.number_input("Respons akustik dalam desibel (30-80dB) (C)", min_value=30.0, max_value=80.0, value=30.0)
D = st.number_input("Massa dalam gram (150-300g) (D)", min_value=150.0, max_value=300.0, value=50.0)
E = st.number_input("Volume dalam sentimeter kubik (100-300cm³) (E)", min_value=100.0, max_value=300.0, value=50.0)

if st.button("Rekomendasikan Avocado"):
    # Preprocessing input
    input_data = np.array([[A,B,C,D,E]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    crop_name = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Rekomendasi Avocado: **{crop_name.upper()}**")
