import streamlit as st
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import time

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = load_model('face_mask_detector.h5', compile=False)

# Custom CSS with more animation
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: Arial, sans-serif;
    }
    h1 {
        text-align: center;
        color: #4CAF50;
        animation: fadeIn 2s ease-in-out;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        animation: bounce 2s infinite;
    }
    .stFileUploader div, .stTextInput div {
        text-align: center;
        animation: pulse 2s infinite;
    }
    .stFileUploader label, .stTextInput label {
        color: #4CAF50;
        font-weight: bold;
        animation: fadeIn 2s ease-in-out;
    }
    .stImage img {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        animation: zoomIn 1s ease-in-out;
    }
    .prediction-result {
        text-align: center;
        color: #4CAF50;
        font-size: 24px;
        font-weight: bold;
        animation: slideIn 1s ease-in-out;
    }
    .confidence {
        text-align: center;
        color: #333;
        font-size: 20px;
        animation: fadeIn 1.5s ease-in-out;
    }
    .loader {
        border: 8px solid #f3f3f3; /* Light grey */
        border-top: 8px solid #4CAF50; /* Green */
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 2s linear infinite;
        margin: auto;
        display: block;
    }
    .sidebar-content {
        animation: slideInRight 1s ease-in-out;
        color: #4CAF50;
        text-align: center;
    }
    .made-by {
        text-align: center;
        color: #4CAF50;
        font-size: 18px;
        margin-top: 20px;
        animation: fadeInUp 2s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-30px);
        }
        60% {
            transform: translateY(-15px);
        }
    }
    @keyframes zoomIn {
        from { transform: scale(0); }
        to { transform: scale(1); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes slideInRight {
        from { transform: translateX(100%); }
        to { transform: translateX(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with description and "Made by" information


st.title('Image Upload and Prediction')

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Input for image URL
image_url = st.text_input('Or enter image URL:')

def load_image_from_file(file):
    img_path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(img_path, 'wb') as f:
        f.write(file.getbuffer())
    img = Image.open(img_path)
    return img

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    img_format = img.format if img.format else 'JPEG'
    img_path = os.path.join(UPLOAD_FOLDER, f"image_from_url.{img_format.lower()}")
    img.save(img_path, format=img_format)
    return img

def preprocess_and_predict(img):
    img = img.resize((128, 128))  # Resize to 128x128
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = img_array / 255.0  # Rescale the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    predictions = model.predict(img_array)
    prediction_result = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return prediction_result, confidence

img = None
if uploaded_file is not None:
    img = load_image_from_file(uploaded_file)
elif image_url:
    img = load_image_from_url(image_url)

if img:
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Loading spinner
    with st.spinner('Making prediction...'):
        time.sleep(2)  # Simulate loading delay
        prediction_result, confidence = preprocess_and_predict(img)

    # Display prediction result with confidence
    if prediction_result == 1:
        st.markdown(f'<div class="prediction-result">The person in the image is wearing a mask</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="prediction-result">The person in the image is not wearing a mask</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="sidebar-content">
        <h2>About this App</h2>
        <p>This application allows you to upload an image either from your computer or via a URL. The uploaded image will be analyzed to determine if the person is wearing a mask or not.</p>
        <div class="made-by">Made by Basanta Dharala</div>
    </div>
    """,
    unsafe_allow_html=True
)