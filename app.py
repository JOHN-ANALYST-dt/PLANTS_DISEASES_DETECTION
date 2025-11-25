import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# --- 1. CONFIGURATION ---
MODEL_PATH = 'mobileNet_model2.h5'
IMG_SIZE = (128,128)
TITLE = "AI-Driven Crop Disease Detector"

# Define the full list of class names (MUST match training order)
FULL_CLASS_NAMES = [
    'Apple Scab', 
    'Apple Black Rot', 
    'Apple Cedar Rust', 
    'Corn Common Rust', 
    'Corn Gray Leaf Spot',
    'Potato Early Blight',
    'Potato Late Blight', 
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Healthy',
    'Pepper Bell Bacterial Spot',
    'Pepper Bell Healthy',
    'Grape Black Rot',
    'Grape Healthy',
    'Cherry Powdery Mildew'
]

# --- Categorization for Tabbed View ---
# We will use this to guide the user on which section their plant belongs to.

VEGETABLE_CLASSES = [
    'Corn Common Rust', 'Corn Gray Leaf Spot',
    'Potato Early Blight', 'Potato Late Blight', 
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Healthy',
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy'
]
FRUIT_CLASSES = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 
    'Grape Black Rot', 'Grape Healthy', 
    'Cherry Powdery Mildew'
]

# --- 2. STREAMLIT PAGE CONFIG (MOVED TO THE TOP) ---
# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title=TITLE, layout="wide")


# --- 3. LOAD MODEL (Original Section 2, Renumbered) ---
@st.cache_resource
def load_trained_model(path):
    """Loads the model from the .h5 file."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: Could not find or load model at '{path}'. Please ensure the file exists. Error: {e}")
        return None

model = load_trained_model(MODEL_PATH)

# --- 4. PREDICTION FUNCTION (Original Section 3, Renumbered) ---
def preprocess_and_predict(img_data, model, class_names, img_size):
    """
    Preprocesses the image data (from PIL or file_uploader) and returns the prediction.
    """
    if model is None:
        return "Model not available.", 0.0, None

    # Ensure img_data is a PIL Image object
    if isinstance(img_data, io.BytesIO):
        img = Image.open(img_data).convert('RGB')
    elif isinstance(img_data, Image.Image):
        img = img_data
    else:
        # Handle the case where the input is neither
        st.error("Invalid image input type.")
        return "Invalid image.", 0.0, None

    # Resize to the model's expected input size
    img = img.resize(img_size)
    
    # Convert PIL Image to Numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale the image array (must match the training pre-processing: 1./255)
    img_array = img_array / 255.0 

    # Make prediction
    predictions = model.predict(img_array)[0]
    
    # Get the predicted class index and confidence
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence, predictions


# --- 5. STREAMLIT APP INTERFACE (Original Section 4, Renumbered) ---

st.markdown(
    f"""
    <style>
    .big-font {{
        font-size:36px !important;
        font-weight: 800;
        color: #008080; /* Teal */
    }}
    .subheader-font {{
        font-size:24px !important;
        color: #4CAF50; /* Green */
        margin-bottom: 20px;
    }}
    </style>
    <div class="big-font">{TITLE}</div>
    <div class="subheader-font">Leveraging MobileNetV2 for Real-Time Crop Disease Diagnosis</div>
    """, 
    unsafe_allow_html=True
)

# Create the two main tabs
tab_vegetables, tab_fruits = st.tabs(["🥕 Vegetable Crops", "🍎 Fruit Crops"])

# --- Helper function to place content within tabs ---
def render_tab_content(crop_type, class_list):
    st.write(f"### 📸 {crop_type} Image Input")
    st.markdown(f"**This detector covers:** {', '.join(sorted(set([c.split(' ')[0] for c in class_list])))}.")
    st.warning("Use the 'Take a Photo' option provided by the camera input when on mobile.")

    # Use the Streamlit Camera Input as the primary method
    camera_img = st.camera_input(f"Take a Photo ({crop_type} Leaf)", key=f"camera_{crop_type}")

    st.markdown("---")
    st.write("### ⬆️ Or Upload a File")
    uploaded_file = st.file_uploader(f"Choose an image from your device ({crop_type} Leaf)...", type=["jpg", "jpeg", "png"], key=f"uploader_{crop_type}")
    
    return camera_img, uploaded_file

# --- Tab Content: Vegetables ---
with tab_vegetables:
    camera_veg, upload_veg = render_tab_content("Vegetable", VEGETABLE_CLASSES)
    current_input = camera_veg if camera_veg is not None else upload_veg

# --- Tab Content: Fruits ---
with tab_fruits:
    camera_fruit, upload_fruit = render_tab_content("Fruit", FRUIT_CLASSES)
    current_input = camera_fruit if camera_fruit is not None else upload_fruit


# --- 6. PREDICTION LOGIC (Consolidated for either tab) ---
input_data = None
# Determine which input was used (only one can be active at a time usually)
if camera_veg is not None:
    input_data = Image.open(camera_veg)
elif upload_veg is not None:
    input_data = upload_veg
elif camera_fruit is not None:
    input_data = Image.open(camera_fruit)
elif upload_fruit is not None:
    input_data = upload_fruit


if input_data is not None:
    st.write("")
    
    # Display the selected image outside the tabs
    with st.container():
        st.subheader("Image Selected for Diagnosis")
        st.image(input_data, caption='Ready for analysis.', use_column_width=True)
    
    # Prediction button
    if st.button('Diagnose Leaf', key='diagnose_button'):
        with st.spinner('Analyzing image for disease...'):
            # Predict using the consolidated input
            predicted_class, confidence, raw_predictions = preprocess_and_predict(
                input_data, model, FULL_CLASS_NAMES, IMG_SIZE
            )

        # --- Display Results ---
        
        st.markdown("## 🔍 Analysis Results")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 🔬 Diagnosis Result")
            
            # Use appropriate styling based on the result
            if 'Healthy' in predicted_class:
                st.success(f"**Status:** {predicted_class}")
                st.balloons()
            else:
                st.error(f"**Disease Detected:** {predicted_class}")
            
            st.info(f"**Confidence:** {confidence*100:.2f}%")
            
            st.markdown("---")
            st.markdown("### 🧑‍🌾 Suggested Action")
            # NOTE: Treatment recommendations should ideally be pulled from a database/config file 
            # associated with the predicted class (like in your PROJECT.ipynb snippets).
            if 'Healthy' in predicted_class:
                 st.write("Keep monitoring the plant. Ensure optimal water and nutrient levels and good air circulation.")
            else:
                 st.write(f"""
                 Immediate intervention is necessary for **{predicted_class}**.
                 Recommended actions often include:
                 1. **Pruning:** Remove and destroy infected leaves/branches.
                 2. **Chemicals:** Apply targeted fungicide (if fungal) or bactericide (if bacterial).
                 3. **Cultural Control:** Improve drainage and spacing.
                 """)
        
        with col2:
            st.markdown("### 📊 Confidence Scores (Top 5)")
            
            # Combine class names and scores and sort for visualization
            class_scores = list(zip(FULL_CLASS_NAMES, raw_predictions))
            class_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Extract top N for bar chart visualization
            top_n = 5
            top_classes = [score[0] for score in class_scores[:top_n]]
            top_confidences = [score[1] for score in class_scores[:top_n]]

            chart_data = {
                'Class': top_classes, 
                'Confidence': top_confidences
            }
            st.dataframe(chart_data)


# --- 7. SIDEBAR INSTRUCTIONS ---
st.sidebar.markdown("### 💡 Current Model Coverage")
st.sidebar.markdown(f"**Vegetables:** {', '.join(sorted(set([c.split(' ')[0] for c in VEGETABLE_CLASSES])))}")
st.sidebar.markdown(f"**Fruits:** {', '.join(sorted(set([c.split(' ')[0] for c in FRUIT_CLASSES])))}")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Deployment Instructions")
st.sidebar.code("""
# 1. Ensure 'best_vegetable_model.h5' is in the same folder
# 2. Run the app:
# streamlit run app.py
""")

st.sidebar.markdown(f"**Trained on:** {len(FULL_CLASS_NAMES)} Disease Classes")
st.sidebar.markdown(f"**Base Model:** MobileNet (or EfficientNetB0)")