import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import numpy as np
import io
from PIL import Image
import os
import pathlib
import base64
import time
# Assuming 'intervention.py' exists in the same directory
from intervention import get_interventions 

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

# Base directory setup
BASE_DIR = pathlib.Path(__file__).parent 

# Paths and Files
MODEL_PATH = os.path.join(BASE_DIR, 'CABBAGE_mobileNet_model7.h5')
BACKGROUND_IMAGE_PATH = os.path.join(BASE_DIR, 'vege2.jpeg') 
CSS_PATH = os.path.join(BASE_DIR, 'style.css')
REJECTION_THRESHOLD = 0.50 # 50% confidence minimum
IMG_SIZE = (248, 248) # Model input size

TITLE = "AgroVision AI: Cabbage Leaf Detector"

CABBAGE_CLASS_NAMES = [
    'cabbage healthy leaf',
    'cabbage black rot',
    'cabbage clubroot',
    'cabbage downy mildew',
    'cabbage black leg (phoma lingam)'
]


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def encode_image_to_base64(path):
    # ... (function body remains unchanged) ...
    # Removed for brevity, but keep the original logic here
    if not os.path.exists(path):
        return "none"
    try:
        ext = os.path.splitext(path)[1].lower()
        mime_type = "image/jpeg" if ext in ('.jpg', '.jpeg') else "image/png"
        with open(path, "rb") as f:
            data = f.read()
            encoded_string = base64.b64encode(data).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        return "none"


def inject_custom_css(file_path):
    """Reads a local CSS file and injects it into the Streamlit app."""
    try:
        # Check for file existence before opening
        abs_path = os.path.join(BASE_DIR, file_path)
        with open(abs_path) as f:
            # st.markdown is the critical command here
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 
    except FileNotFoundError:
        st.warning(f"CSS file not found at path: {file_path}. Using default Streamlit styling.")
    except Exception as e:
        st.error(f"Error injecting CSS: {e}")
# REMOVED: inject_custom_css("style.css") # This was the mistake!

@st.cache_resource
def load_trained_model(path):
    # ... (function body remains unchanged) ...
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: Could not load model at '{path}'. Error: {e}")
        return None

def preprocess_and_predict(img_data, model, class_names, img_size):
    # ... (function body remains unchanged) ...
    pass


# ==============================================================================
# 3. APP EXECUTION START
# ==============================================================================

# --- Setup (MUST be the first executable st commands) ---
st.set_page_config(page_title=TITLE, layout="centered") # FIXED: This must be FIRST!

# Now that config is set, we can run other st commands within functions
inject_custom_css(CSS_PATH) 
model = load_trained_model(MODEL_PATH)


# --- UI: Title and Info ---
st.markdown(
    f"""
    <div class="title-container">
        <div class="big-font">{TITLE}</div>
        <div class="subheader-font">Real Time Crop Disease Diagnosis</div>
    </div>
    """, 
    unsafe_allow_html=True
)

st.info("This application is specialized for detecting the following **Cabbage** issues: " + ', '.join([c.replace('cabbage ', '') for c in CABBAGE_CLASS_NAMES]))

# ... (The rest of the application code remains unchanged and should now work) ...