import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
import os

# Download the model if it's not already in the directory
@st.cache(allow_output_mutation=True)
def download_model(url):
    if not os.path.exists('model.h5'):
        r = requests.get(url, allow_redirects=True)
        with open('model.h5', 'wb') as f:
            f.write(r.content)

# URL where the model is hosted (use your own URL)
model_url = 'D:\HTML\plant_detection\model.h5'
download_model(model_url)

# Load the model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('model.h5')
    return model

model = load_trained_model()

# Preprocess the uploaded image
def preprocess_image(image, target_size=(225, 225)):
    img = image.resize(target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    return x

# Streamlit app layout
st.title('Leaf Disease Classification')
st.write("Upload a leaf image to classify it as Healthy, Powdery, or Rusty.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        x = preprocess_image(image)
        prediction = model.predict(x)
        labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
        predicted_label = labels[np.argmax(prediction)]
        st.write(f'Prediction: {predicted_label}')

# Example URL for downloading the model
model_url = 'D:\HTML\plant_detection\model.h5'
download_model(model_url)
