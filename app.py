import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

@st.cache_resource
def load_model():
    base_model = VGG16(input_shape=(224, 224, 3), weights=None, include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    model.load_weights("waste.h5")
    return model
# Streamlit UI
st.title("Waste Classification App")
st.write("Upload an image to classify it into different waste categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array)
    class_names = ["batteries", "clothes", "e-waste", "glass", "light blubs", "metal", "organic", "paper", "plastic"]  
    predicted_class = class_names[np.argmax(predictions)]
    
    st.write(f"Prediction: **{predicted_class}**")
