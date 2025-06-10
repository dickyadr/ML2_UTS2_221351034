import streamlit as st
import tensorflow as tf
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
st.title("Prediksi Kematangan Avocado")
st.write("Masukkan informasi karakteristik Avocado:")

# Form input pengguna (5 fitur sesuai model)
A = st.number_input("Fitur A: Kemurnian warna (%)", 0.0, 100.0, 14.5)
B = st.number_input("Fitur B: Intensitas cahaya (%)", 0.0, 100.0, 19.0)
C = st.number_input("Fitur C: Respons akustik (30–80 dB)", 30.0, 80.0, 40.0)
D = st.number_input("Fitur D: Massa (gram)", 150.0, 300.0, 175.0)
E = st.number_input("Fitur E: Volume (cm³)", 100.0, 300.0, 261.0)

# Tombol Prediksi
if st.button("Prediksi Kematangan"):
    try:
        # Preprocessing input
        input_data = np.array([[A, B, C, D, E]])
        input_scaled = scaler.transform(input_data).astype(np.float32)

        # Inference dengan TFLite
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_label = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_label])[0]

        st.success(f"Avocado diprediksi sebagai: **{predicted_class.upper()}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
