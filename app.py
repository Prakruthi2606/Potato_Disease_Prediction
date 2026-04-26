import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("potato_model.h5")

# Class names
class_names = ['Early_Blight', 'Late_Blight', 'Healthy']

st.title("🥔 Potato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((128,128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"### Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
