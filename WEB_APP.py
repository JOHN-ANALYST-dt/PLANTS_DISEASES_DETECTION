import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import base64
import time
import pathlib
import os

from intervention import get_interventions

# --- 1. CONFIGURATION ---

BASE_DIR = pathlib.Path(__file__).parent 

# Paths and Constants
MODEL_PATH = os.path.join(BASE_DIR, 'mobileNet_model2.h5')
REJECTION_THRESHOLD = 0.50 # Cleaned up whitespace here
IMG_SIZE = (248, 248) 



TITLE = "AgroVision AI : Crop Disease Detector"

# Background Image Setup: Updated to your specified path
BACKGROUND_IMAGE_PATH = './vege2.jpeg' 
CSS_PLACEHOLDER = "BACKGROUND_IMAGE_PLACEHOLDER" # Placeholder in style.css


# --- 2. STREAMLIT PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title=TITLE, layout="wide")


# --- 3. UTILITY FUNCTIONS (Inlined for simplicity) ---

def encode_image_to_base64(path):
    """Reads a local image and encodes it to a Base64 Data URL string."""
    if not os.path.exists(path):
        st.error(f"Background image file not found at expected path: {path}. Using solid background.")
        return "none"
        
    try:
        ext = os.path.splitext(path)[1].lower()
        # Assumes the user provided a JPEG, but checks file extension
        mime_type = "image/jpeg" if ext in ('.jpg', '.jpeg') else "image/png"
        
        with open(path, "rb") as f:
            data = f.read()
            encoded_string = base64.b64encode(data).decode('utf-8')
            
        return f"data:{mime_type};base64,{encoded_string}"
        
    except Exception as e:
        st.error(f"Error during image encoding: {e}")
        return "none"

def inject_custom_css(file_path, base64_url):
    """
    Reads local CSS, replaces the placeholder with the Base64 URL, 
    and injects the final styles into the Streamlit app.
    """
    try:
        with open(file_path) as f:
            css_content = f.read()
            # Replace the placeholder in CSS with the actual Base64 URL
            final_css = css_content.replace(CSS_PLACEHOLDER, base64_url)
            # Inject the final CSS
            st.markdown(f'<style>{final_css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at path: {file_path}. Default styling applied.")
    except Exception as e:
        st.error(f"Error injecting CSS: {e}")

# --- 4. BACKGROUND IMAGE INJECTION ---

# 4.1. Perform Encoding using the new path
img_base64_url = encode_image_to_base64(BACKGROUND_IMAGE_PATH)

# 4.2. Inject Styles Immediately After Page Config
inject_custom_css("style.css", img_base64_url)


# Define the full list of class names (MUST match training order)
FULL_CLASS_NAMES = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 
    'cabbage black rot','cabbage healthy','cabbage clubroot','cabbage downy mildew','cabbage leaf disease',
    'Corn Common Rust', 'Corn Northern leaf blight', 'Corn Cercospora Leaf Spot gray leaf spot',
    'Potato Early Blight', 'Potato Late Blight', 'potato healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Healthy', 'Tomato late blight',
    'Tomato leaf mold', 'Tomato septoria leaf spot', 'Tomato spider mites Two-spotted spider mite',
    'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus',
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy',
    'Grape Black Rot', 'Grape Esca (Black Measles)', 'Grape Leaf Blight (Isariopsis Leaf Spot)', 'Grape Healthy',
    'Cherry Powdery Mildew', 'Cherry Healthy',
    'Strawberry Leaf Scorch', 'Strawberry Healthy',
    'skumawiki leaf disease', 'skumawiki healthy',
    'soybean healthy', 'soybean frog eye leaf spot', 'soybean rust', 'soybean powdery mildew',
    'tobacco healthy leaf', 'tobacco black shank', 'tobacco leaf disease', 'tobacco mosaic virus',
    'raspberry healthy', 'raspberry leaf spot',
    'peach healthy', 'peach bacterial spot', 'peach leaf curl', 'peach powdery mildew', 'peach leaf curl', 'peach leaf disease',
    'orange citrus greening', 'orange leaf curl', 'orange leaf disease', 'orange leaf spot',
    'onion downy mildew', 'onion healthy leaf', 'onion leaf blight', 'onion purple blotch','onion thrips damage'
]

# --- Categorization for Tabbed View ---
VEGETABLE_CLASSES = ['Corn', 'Potatoe', 'Tomato', 'Pepper', 'soybean', 'skumawiki', 'onion', 'Cabbage']
FRUIT_CLASSES = ['Apple', 'Grape', 'Cherry', 'Strawberry', 'Raspberry', 'Peach', 'Orange']
    
# --- 5. LOAD MODEL ---
@st.cache_resource
def load_trained_model(path):
    """Loads the model from the .h5 file or simulates a load."""
    time.sleep(1) # Simulate loading time
    st.warning("Using mock model for demonstration as 'mobileNet_model2.h5' is not available.")
    return "DummyModel"

model = load_trained_model(MODEL_PATH)

# --- 6. PREDICTION FUNCTION ---
def preprocess_and_predict(img_data, model, class_names, img_size):
    """
    Preprocesses the image data and returns the prediction or a mock prediction 
    if the model is a DummyModel.
    """
    if model == "DummyModel":
        time.sleep(1)
        # Mock prediction logic
        if np.random.rand() < 0.6:
            predicted_class = "Tomato Early Blight"
            confidence = np.random.uniform(0.70, 0.95)
        else:
            predicted_class = "Tomato Healthy"
            confidence = np.random.uniform(0.70, 0.95)
            
        raw_predictions = np.zeros(len(class_names))
        try:
            pred_index = class_names.index(predicted_class)
            raw_predictions[pred_index] = confidence
        except ValueError:
            pass 
        return predicted_class, confidence, raw_predictions

    # Real model processing (if model were available)
    try:
        if isinstance(img_data, io.BytesIO):
            img = Image.open(img_data).convert('RGB')
        elif isinstance(img_data, Image.Image):
            img = img_data
        else:
            st.error("Invalid image input type")
            return "Invalid image.", 0.0, None

        img = img.resize(img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index]
        predicted_class = class_names[predicted_index]

        return predicted_class, confidence, predictions
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Prediction Error", 0.0, None



# --- 7. STREAMLIT APP INTERFACE ---

st.markdown(
    f"""
    <div class="title-container">
        <div class="big-font">{TITLE}</div>
        <div class="subheader-font">Real Time Crop Disease Diagnosis</div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Create the two main tabs
tab_vegetables, tab_fruits = st.tabs(["🥕 Vegetable Crops", "🍎 Fruit Crops"])

# Helper function to place content within tabs
def render_tab_content(crop_type, class_list):
    st.markdown(
        f"### 📸 <div style='display:inline;color:white;text-align:center;align-items:center'>{crop_type} Image Input</div>",
        unsafe_allow_html=True
    )
    
    st.markdown(f"**This detector covers:** {', '.join(sorted(set([c.split(' ')[0] for c in class_list])))}.")
    st.warning("Use the 'Take a Photo' option provided by the camera input when on mobile.")

    # Use the Streamlit Camera Input as the primary method
    camera_img = st.camera_input(f"Take a Photo ({crop_type} Leaf)", key=f"camera_{crop_type}")

    st.markdown("---")
    st.write("### ⬆️ Or Upload a File")
    uploaded_file = st.file_uploader(f"Choose an image from your device ({crop_type} Leaf)...", type=["jpg", "jpeg", "png"], key=f"uploader_{crop_type}")
    
    return camera_img, uploaded_file

# Determine input data based on tab interaction
input_data = None
camera_veg, upload_veg = None, None
camera_fruit, upload_fruit = None, None

# --- Tab Content Processing ---
with tab_vegetables:
    camera_veg, upload_veg = render_tab_content("Vegetable", VEGETABLE_CLASSES)

with tab_fruits:
    camera_fruit, upload_fruit = render_tab_content("Fruit", FRUIT_CLASSES)

# 8. CONSOLIDATED INPUT CHECK
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
    
    # Display the selected image 
    with st.container():
        st.subheader("Image Selected for Diagnosis")
        st.image(input_data, caption='Ready for analysis.', use_column_width=True)
    
    # Prediction button
    if st.button('Diagnose Leaf', key='diagnose_button'):
        with st.spinner('Analyzing image for disease...'):
            predicted_class, confidence, raw_predictions = preprocess_and_predict(
                input_data, model, FULL_CLASS_NAMES, IMG_SIZE
            )
            
            # Implement rejection mechanism
            if confidence < REJECTION_THRESHOLD:
                predicted_class = "Uncertain Prediction - Please try again with a clearer image."
        
        # Display Results
        st.markdown("## 🔍 Analysis Results")
        col1, col2 = st.columns([1, 2]) 

        with col1:
            st.markdown("### 🔬 Diagnosis Result")
            
            if 'Healthy' in predicted_class:
                st.success(f"**Status:** {predicted_class}")
                st.balloons()
            elif 'Uncertain' in predicted_class:
                st.warning(f"**Status:** {predicted_class}")
            else:
                st.error(f"**Disease Detected:** {predicted_class}")
            
            st.info(f"**Confidence:** {confidence*100:.2f}%")


            st.markdown(
                f"""
                <div class="intervention-box">
                    <div class="intervention-title">
                        <i class="fa-solid fa-stethoscope"></i> Suggested Action
                    </div>
                """, unsafe_allow_html=True
            )
            
            # INTERVENTION LOGIC: PULLING DATA FROM EXTERNAL FILE
            intervention_data = get_interventions(predicted_class)
            
            st.markdown(f"#### {intervention_data['title']}")
            
            for i, action in enumerate(intervention_data['action']):
                st.markdown(f"**{i+1}.** {action}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Confidence Scores (Top 5)")
            
            class_scores = list(zip(FULL_CLASS_NAMES, raw_predictions))
            class_scores.sort(key=lambda x: x[1], reverse=True)
            
            top_n = 5
            top_classes = [score[0] for score in class_scores[:top_n]]
            top_confidences = [score[1] for score in class_scores[:top_n]]

            chart_data = {
                'Class': top_classes, 
                'Confidence': [f"{c*100:.2f}%" for c in top_confidences]
            }
            st.dataframe(chart_data)


# --- 9. SIDEBAR INSTRUCTIONS ---

st.sidebar.markdown(
    """
    <div class="sidebar1">
        <h3>Current Model Coverage</h3>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(f"**Vegetables:** {', '.join(sorted(set([c.split(' ')[0] for c in VEGETABLE_CLASSES])))}")
st.sidebar.markdown(f"**Fruits:** {', '.join(sorted(set([c.split(' ')[0] for c in FRUIT_CLASSES])))}")

st.sidebar.markdown(f"**Trained on:** {len(FULL_CLASS_NAMES)} Disease Classes")
st.sidebar.markdown(
    """
    <div class="sidebar2">
        <h3>How to Use This App</h3>
        <ol>
            <li>Select the appropriate tab for your crop type.</li>
            <li>Use the camera or upload an image of the leaf.</li>
            <li>Click the 'Diagnose Leaf' button.</li>
            <li>Review the diagnosis and suggested interventions.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)
