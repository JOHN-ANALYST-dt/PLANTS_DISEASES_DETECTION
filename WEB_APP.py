import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os 
import pathlib

# --- 1. CONFIGURATION ---

BASE_DIR = pathlib.Path(__file__).parent 

# 2. Construct the full path: Joins the base directory, the folder name ('MODELS'), 
# and the file name to create an absolute path.
MODEL_PATH = os.path.join(BASE_DIR, 'MODELS', 'inceptionv3_model2.h5')
REJECTION_THRESHOLD = 0.80

IMG_SIZE = (128, 128) # Model input size
TITLE = "AgroVision AI : Crop Disease Detector"

# --- URL for the title background image (REPLACE WITH YOUR OWN) ---
# NOTE: Using a high-quality, free stock image URL as a placeholder.
BACKGROUND_IMAGE_URL = 'https://images.unsplash.com/photo-1542838132-92c24c965c40?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8fGVufDB8fHx8fA%3D%3D'
# --- URL for the sidebar background image (A lighter, subtle texture) ---
SIDEBAR_IMAGE_URL = 'https://images.unsplash.com/photo-1506869640319-fe1a24fd767c?q=80&w=1780&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'


# Define the full list of class names (MUST match training order)
FULL_CLASS_NAMES = [
    'Apple Scab', 
    'Apple Black Rot', 
    'Apple Cedar Rust', 
    'cabbage black rot','cabbage healthy','cabbage clubroot','cabbage downy mildew','cabbage leaf disease',
    'Corn Common Rust', 
    'Corn Northern leaf blight',
    'Corn Cercospora Leaf Spot gray leaf spot',
    'Potato Early Blight',
    'Potato Late Blight', 
    'potato healthy',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Healthy',
    'Tomato late blight',
    'Tomato leaf mold',
    'Tomato septoria leaf spot',
    'Tomato spider mites Two-spotted spider mite',
    'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato mosaic virus',
    'Pepper Bell Bacterial Spot',
    'Pepper Bell Healthy',
    'Grape Black Rot',
    'Grape Esca (Black Measles)',
    'Grape Leaf Blight (Isariopsis Leaf Spot)',
    'Grape Healthy',
    'Cherry Powdery Mildew',
    'Cherry Healthy',
    'Strawberry Leaf Scorch',
    'Strawberry Healthy',
    'skumawiki leaf disease',
    'skumawiki healthy',
    'soybean healthy',
    'soybean frog eye leaf spot',
    'soybean rust',
    'soybean powdery mildew',
    'tobacco healthy leaf',
    'tobacco black shank',
    'tobacco leaf disease',
    'tobacco mosaic virus',
    'raspberry healthy',
    'raspberry leaf spot',
    'peach healthy',
    'peach bacterial spot',
    'peach leaf curl',
    'peach powdery mildew',
    'peach leaf curl',
    'peach leaf disease',
    'orange citrus greening',
    'orange leaf curl',
    'orange leaf disease',
    'orange leaf spot',
    'onion downy mildew',
    'onion healthy leaf',
    'onion leaf blight',
    'onion purple blotch','onion thrips damage'
]

# --- Categorization for Tabbed View ---
VEGETABLE_CLASSES = [
'Corn', 'Potatoe', 'Tomatoe', 'Pepper', 'soybean', 'skumawiki', 'onion', 'Cabbage'
]
FRUIT_CLASSES = [
'Apple', 'Grape', 'Cherry', 'Strawberry', 'Raspberry', 'Peach', 'Orange'
]
    

# --- 2. STREAMLIT PAGE CONFIG (MOVED TO THE TOP) ---
st.set_page_config(page_title=TITLE, layout="wide")


# --- 3. LOAD MODEL (Original Section 2, Renumbered) ---
@st.cache_resource
def load_trained_model(path):
    """Loads the model from the .h5 file."""
    try:
        # Suppress TF warnings during loading
        with tf.get_logger().disable_resource_variables():
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
        st.error("Invalid image input type")
        return "Invalid image.", 0.0, None

    # Resize to the model's expected input size
    img = img.resize(img_size)
    
    # Convert PIL Image to Numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the input shape (1, 128, 128, 3)
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
    /* ---------------------------------------------------- */
    /* 1. Global Centering and Layout Customization (CENTER THE CONTENT) */
    /* Applies a max-width and centers the main content block */
    .stApp {{
        margin: 0 auto !important;
        max-width: 1200px; /* Adjust as needed */
    }}
    
    /* 2. Sidebar Styling (GRADIENT COLOR WITH PICTURES) */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(76, 175, 80, 0.9), rgba(0, 128, 128, 0.9)); /* Green to Teal Gradient */
        background-image: url("{SIDEBAR_IMAGE_URL}");
        background-size: cover;
        background-blend-mode: multiply; /* Blend gradient with image */
        color: white; 
    }}
    /* Ensure sidebar text and links are visible */
    [data-testid="stSidebar"] .stMarkdown h3, [data-testid="stSidebar"] .stMarkdown p {{
        color: white !important;
        text-shadow: 1px 1px 2px #000000;
    }}

    /* 3. Title Background (BACKGROUND PICTURE OF A PLANTS OF VEGETABLES) */
    /* Target the main header container */
    [data-testid="stVerticalBlock"] > div:nth-child(1) {{
        background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url("{BACKGROUND_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        padding: 40px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }}

    /* 4. Font Styling (Ensure contrast against background) */
    .big-font {{
        font-size: 36px !important;
        font-weight: 900;
        color: white; /* White for contrast */
        text-shadow: 2px 2px 4px #000000;
    }}
    .subheader-font {{
        font-size: 24px !important;
        color: #C8E6C9; /* Lighter green for subheader */
        margin-bottom: 0px;
        text-shadow: 1px 1px 2px #000000;
    }}
    
    /* 5. Responsive Buttons (Full Width Diagnose Button) */
    /* Target the diagnose button directly by key */
    button[data-testid="baseButton-secondary"] {{
        width: 100%; /* Make button full width */
        font-size: 1.2rem;
        font-weight: bold;
        padding: 15px 10px;
        background-color: #008080; /* Teal color */
        color: white;
        border: none;
        transition: 0.3s;
    }}
    button[data-testid="baseButton-secondary"]:hover {{
        background-color: #4CAF50; /* Green hover */
    }}
    
    /* 6. Camera Input Styling (Icon Look) */
    /* Hides the default label of st.camera_input */
    div[data-testid="stCameraInput"] > label {{
        display: none !important;
    }}
    /* Style the camera button to look like a styled, full-width button */
    div[data-testid="stCameraInput"] > div:first-child > button {{
        background-color: #f4a261; /* Orange for capture action */
        color: white;
        padding: 10px 20px;
        font-size: 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        display: block; 
        width: 100%;
        font-weight: bold;
        margin-bottom: 10px;
    }}
    </style>
    <div class="big-font">{TITLE}</div>
    <div class="subheader-font">Real Time Crop Disease Diagnosis</div>
    """, 
    unsafe_allow_html=True
)


# Create the two main tabs
tab_vegetables, tab_fruits = st.tabs(["🥕 Vegetable Crops", "🍎 Fruit Crops"])

# Helper function to place content within tabs
def render_tab_content(crop_type, class_list):
    st.write(f"### 📸 {crop_type} Image Input")
    st.markdown(f"**This detector covers:** {', '.join(sorted(set([c.split(' ')[0] for c in class_list])))}.")
    
    # Styled camera input (uses CSS for icon look)
    camera_img = st.camera_input("📸 Click to Open Camera", key=f"camera_{crop_type}") 
    st.warning("Click the camera button above to take a photo. (Only still photos are captured).")

    st.markdown("---")
    st.write("### ⬆️ Or Upload a File")
    uploaded_file = st.file_uploader(f"Choose an image from your device ({crop_type} Leaf)...", type=["jpg", "jpeg", "png"], key=f"uploader_{crop_type}")
    
    return camera_img, uploaded_file

# Tab Content: Vegetables 
with tab_vegetables:
    camera_veg, upload_veg = render_tab_content("Vegetable", VEGETABLE_CLASSES)
    current_input = camera_veg if camera_veg is not None else upload_veg

# Tab Content: Fruits 
with tab_fruits:
    camera_fruit, upload_fruit = render_tab_content("Fruit", FRUIT_CLASSES)
    current_input = camera_fruit if camera_fruit is not None else upload_fruit


# 6. PREDICTION LOGIC (Consolidated for either tab) 
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
            
            # VALIDATION AND REJECTION CHECK
            if confidence < REJECTION_THRESHOLD:
                # 🚨 Low confidence suggests the image is not a plant leaf
                st.warning("⚠️ **Low Confidence Prediction.**")
                st.error(f"Prediction confidence ({confidence*100:.2f}%) is below the {REJECTION_THRESHOLD*100:.0f}% threshold.")
                st.markdown("### **Only plant leaves accepted.** Please upload a clearer image of a single leaf.")
                return
            
        # Display Results
        st.markdown("## 🔍 Analysis Results")
        col1, col2 = st.columns([1, 2]) 

        with col1:
            st.markdown("### 🔬 Diagnosis Result")
            
            # Use appropriate styling based on the result
            if 'Healthy' in predicted_class:
                # Customized success message (pop-up)
                st.success(f"**Status: HEALTHY!** The leaf is free of common diseases. ({predicted_class})")
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
                3. **Cultural Control:** Improve drainage and spacing
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
st.sidebar.markdown("### Current Model Coverage")
st.sidebar.markdown(f"**Vegetables:** {', '.join(sorted(set([c.split(' ')[0] for c in VEGETABLE_CLASSES])))}")
st.sidebar.markdown(f"**Fruits:** {', '.join(sorted(set([c.split(' ')[0] for c in FRUIT_CLASSES])))}")


st.sidebar.markdown(f"**Trained on:** {len(FULL_CLASS_NAMES)} Disease Classes")
st.sidebar.markdown(f"**Base Model:** InceptionV3 (Model input size: {IMG_SIZE[0]}x{IMG_SIZE[1]})")