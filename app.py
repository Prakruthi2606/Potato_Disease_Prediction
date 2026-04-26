import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Potato Plant Disease AI", layout="wide")

# 🌿 CSS
st.markdown("""
<style>

/* Header */
.header {
    background: linear-gradient(135deg, #1b5e20, #66bb6a);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 20px;
}

.header h1 {
    color: black;
    font-size: 36px;
}

.header p {
    color: #102a12;
}

/* Upload */
.upload {
    background-color: #e8dcc6;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    padding: 10px 18px;   /* smaller height + width */
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-top: 15px;
    color: #f5f5f5 !important;
    display: inline-block;   /* makes box fit content */
}

/* Confidence card */
.conf-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    color: #2e2e2e;
    text-align: center;
}

/* Big % */
.percentage {
    font-size: 36px;
    font-weight: 800;
    color: #2e7d32;
    margin-top: 10px;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
}

</style>
""", unsafe_allow_html=True)

# Load model
model = load_model("potato_model.h5")
class_names = ['Early_Blight', 'Late_Blight', 'Healthy']

# Header
st.markdown("""
<div class="header">
    <h1>🥔🌿 Potato Leaf Disease Detection</h1>
    <p>Upload a potato leaf image to detect plant health</p>
</div>
""", unsafe_allow_html=True)

# Upload
st.markdown('<div class="upload">📤 Upload Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns([1.2, 1])

    img = Image.open(uploaded_file)

    # Preprocess
    img_resized = img.resize((128,128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Label
    if predicted_class == "Healthy":
        label = "Healthy 🌿"
        emoji = "✅"
        pred_text = f"Prediction: {label}"
    elif predicted_class == "Early_Blight":
        label = "Early Blight 🍂"
        emoji = "⚠️"
        pred_text = f"⚠️ Prediction: {label}"
    else:
        pred_text = f"⚠️ Prediction:"
        label = "Late Blight 🦠"
        emoji = "🚨"

    # LEFT → Image + Prediction BELOW
    with col1:
        st.image(img, caption="🌱 Uploaded Image", use_container_width=True)

        st.markdown(
            f'''
            <div class="result-card">
                <span style="font-size:16px; font-weight:500;">{pred_text}</span>
                {emoji} {label}
            </div>
            ''',
            unsafe_allow_html=True
        )

    # RIGHT → ONLY Confidence
    with col2:
        st.markdown('<div class="conf-card">', unsafe_allow_html=True)

        st.markdown("### 📊 Confidence Level")

        st.markdown(f"""
<div style="
    background: #e0e0e0;
    border-radius: 12px;
    height: 25px;
    width: 100%;
    position: relative;
    overflow: hidden;
">
    <div style="
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        width: {confidence*100}%;
        height: 100%;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    ">
        {confidence*100:.0f}%
    </div>
</div>
""", unsafe_allow_html=True)

       

        # ✅ Added confidence text label
        st.markdown(
            f"<div style='font-size:16px;'>Confidence: {confidence*100:.0f}%</div>",
            unsafe_allow_html=True
        )

        # Confidence status
        if confidence > 0.9:
            status = "🔥 High Confidence"
        elif confidence > 0.7:
            status = "⚡ Medium Confidence"
        else:
            status = "⚠️ Low Confidence"

        st.success(status)

        st.markdown('</div>', unsafe_allow_html=True)
