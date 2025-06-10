import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load model dan tools
model = tf.keras.models.load_model("avocado_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("Prediksi Kematangan Alpukat 🥑")

st.markdown("Masukkan nilai-nilai fitur alpukat di bawah untuk memprediksi tingkat kematangannya:")

# Input fitur (harus sesuai urutan fiturnya)
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)
f5 = st.number_input("Feature 5", value=0.0)
f6 = st.number_input("Feature 6", value=0.0)
f7 = st.number_input("Feature 7", value=0.0)
f8 = st.number_input("Feature 8", value=0.0)

input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    # Normalisasi
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    pred_class = np.argmax(prediction)
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    st.success(f"Tingkat kematangan alpukat: **{pred_label}**")