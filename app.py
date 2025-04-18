import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("water_pollution_model.h5")

# Define class names
class_names = ['Safe', 'Moderate', 'Dangerous']

st.title("🌊 Water Pollution Classifier")
st.write("Upload an image of a water body and let the AI tell you if it's safe, moderately polluted, or dangerous.")

# Upload image
uploaded_file = st.file_uploader("Choose a water image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))  # match training size
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}** 🌱")
