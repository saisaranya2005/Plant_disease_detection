import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the model from a local file
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model_path = 'D:/HTML/plant_detection/model.h5'
    model = load_model(model_path)
    return model

model = load_trained_model()

# Title of the app
st.title('Plant Disease Detection')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    from PIL import Image
    img = Image.open(uploaded_file)
    img = img.resize((225, 225))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    if st.button('Predict'):
        prediction = model.predict(img)
        labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
        predicted_label = labels[np.argmax(prediction)]
        st.write(f'Prediction: {predicted_label}')

# Function to download the model from a URL if it's too large for GitHub
@st.cache(allow_output_mutation=True)
def download_model(url):
    import requests
    import os

    if not os.path.exists('model.h5'):
        r = requests.get(url, allow_redirects=True)
        open('model.h5', 'wb').write(r.content)

# Example URL for downloading the model (this won't be used if running locally)
# model_url = 'https://your_model_download_link'
# download_model(model_url)
