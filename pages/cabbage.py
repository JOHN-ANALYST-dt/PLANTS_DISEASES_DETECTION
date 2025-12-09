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

# Model Parameters
REJECTION_THRESHOLD = 0.50 # 50% confidence minimum
IMG_SIZE = (248, 248) # Model input size
CABBAGE_CLASS_NAMES = [
    'cabbage healthy leaf',
    'cabbage black rot',
    'cabbage clubroot',
    'cabbage downy mildew',
    'cabbage black leg (phoma lingam)'
]

# App Display
TITLE = "AgroVision AI: Cabbage Leaf Detector"


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def encode_image_to_base64(path):
    """Reads a local image and encodes it to a Base64 Data URL string."""
    if not os.path.exists(path):
        # st.error(f"Background image file not found at expected path: {path}. Using solid background.")
        return "none"
        
    try:
        ext = os.path.splitext(path)[1].lower()
        mime_type = "image/jpeg" if ext in ('.jpg', '.jpeg') else "image/png"
        
        with open(path, "rb") as f:
            data = f.read()
            encoded_string = base64.b64encode(data).decode('utf-8')
            
        return f"data:{mime_type};base64,{encoded_string}"
        
    except Exception as e:
        # st.error(f"Error during image encoding: {e}")
        return "none"

def inject_custom_css(file_path):
    """Reads a local CSS file and injects it into the Streamlit app."""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at path: {file_path}. Using default Streamlit styling.")
    except Exception as e:
        st.error(f"Error injecting CSS: {e}")
inject_custom_css("style.css")         

@st.cache_resource
def load_trained_model(path):
    """Loads the model from the .h5 file."""
    try:
        # Load the model, specifying custom objects if necessary
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: Could not load model at '{path}'. Error: {e}")
        return None

def preprocess_and_predict(img_data, model, class_names, img_size):
    """
    Preprocesses the image data and returns the prediction result dictionary.
    Includes MobileNetV2-specific preprocessing.
    """
    if model is None:
        return {"status": "error", "diagnosis": "Model Not Available", "message": "The AI model failed to load."}

    try:
        # --- Image Loading and Resizing ---
        if isinstance(img_data, Image.Image):
            img = img_data
        elif hasattr(img_data, 'getvalue'): # Streamlit uploaded file object
            img = Image.open(io.BytesIO(img_data.getvalue())).convert('RGB')
        else:
            raise ValueError("Invalid image input type")

        img = img.resize(img_size)
        
        # --- Preprocessing ---
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, 3)
        img_array = mobilenet_preprocess(img_array) # MobileNetV2 normalization [-1, 1]

        # --- Prediction ---
        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index]
        predicted_class = class_names[predicted_index]
        
        
        # --- RESULT LOGIC ---
        
        # 1. Low Confidence/Rejection Logic
        if confidence < REJECTION_THRESHOLD:
            return {
                "status": "inconclusive", 
                "diagnosis": "Diagnosis Inconclusive", 
                "confidence": confidence,
                "class": predicted_class,
                "message": (
                    f"The model is not confident in its top prediction ({confidence*100:.2f}% < {REJECTION_THRESHOLD*100:.0f}%). "
                    "Please try again with a clearer image, closer view of the leaf, or better lighting."
                ),
                "raw_predictions": predictions
            }
        
        # 2. Healthy Leaf
        elif 'healthy' in predicted_class.lower():
            return {
                "status": "healthy",
                "diagnosis": "Excellent News! Healthy Cabbage Leaf Detected",
                "confidence": confidence,
                "class": predicted_class,
                "message": (
                    f"Your cabbage leaf appears **healthy and robust!** "
                    f"Confidence: {confidence*100:.2f}%. Ensure adequate calcium and boron supply "
                    "to prevent tip burn and maintain compact head formation."
                ),
                "raw_predictions": predictions
            }

        # 3. Disease Detected
        else:
            return {
                "status": "warning",
                "diagnosis": f"Disease Detected: {predicted_class}",
                "confidence": confidence,
                "class": predicted_class,
                "message": f"Immediate action is required! The likely issue is **{predicted_class}**.",
                "raw_predictions": predictions
            }

    except Exception as e:
        # st.exception(e) # Keep this in a real app for detailed debugging
        return {"status": "error", "diagnosis": "Processing Error", "message": f"An unexpected error occurred during image processing."}


# ==============================================================================
# 3. APP EXECUTION START
# ==============================================================================

# --- Setup ---
st.set_page_config(page_title=TITLE, layout="centered")
inject_custom_css(CSS_PATH) # Inject CSS immediately
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

# --- UI: Input Section ---
st.markdown("### 📸 Image Input")
st.warning("For best results, take a clear photo of the *affected area* of a single leaf, or upload a high-quality image.")

col_cam, col_upload = st.columns(2)
camera_img = col_cam.camera_input("1. Take a Photo of the Leaf")
uploaded_file = col_upload.file_uploader("2. Upload an Image from Device", type=["jpg", "jpeg", "png"])

# --- Logic: Determine Input and Execute Prediction ---
input_data = None
if camera_img is not None:
    # Convert camera input to PIL Image immediately for consistent handling
    input_data = Image.open(camera_img) 
elif uploaded_file is not None:
    input_data = uploaded_file

if input_data is not None:
    st.markdown("---")
    st.subheader("Image Selected for Analysis")
    
    image_col, result_col = st.columns([1, 1])

    with image_col:
        # Display the image once input is ready
        st.image(input_data, caption='Ready for analysis.', use_column_width=True)
    
    with result_col:
        # Prediction button
        if st.button('Diagnose Leaf Now', key='diagnose_button', use_container_width=True):
            
            with st.spinner('Running specialized Cabbage leaf analysis...'):
                results = preprocess_and_predict(input_data, model, CABBAGE_CLASS_NAMES, IMG_SIZE)
            
            st.markdown("### 🔬 Diagnosis Result")
            
            # --- Displaying Results based on Status ---
            if results["status"] == "healthy":
                st.success(f"**Status:** {results['class']}")
                st.balloons()
                # Simplified HTML structure for the healthy message
                st.markdown(
                    f"""
                    <div class="healthy-prompt">
                        <div class="emoji">🥬🌿</div>
                        <p class="message">{results['message']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
            elif results["status"] == "warning":
                st.error(f"**Disease Detected:** {results['class']}")
                
                # Fetch and display interventions
                intervention_data = get_interventions(results['class']) 
                st.markdown(
                    f"""
                    <div class="intervention-box">
                        <div class="intervention-title">
                            ⚠️ Immediate Action Required
                        </div>
                        <p class="mb-2">{results['message']}</p>
                        <h4 class='text-lg font-bold text-red-700 mt-3'>Recommended Treatment ({intervention_data['title']}):</h4>
                    </div>
                    """, unsafe_allow_html=True
                )
                # Display actions in a structured list
                for i, action in enumerate(intervention_data['action']):
                    st.markdown(f"**{i+1}.** {action}")

            elif results["status"] == "inconclusive":
                st.warning(f"**Status:** {results['diagnosis']}")
                st.info(results['message'])

            # --- Display Top Predictions (for all non-error results) ---
            if results["status"] != "error":
                st.markdown("---")
                st.markdown("### 📊 Top Prediction Scores")
                st.markdown(f"**Confidence:** {results['confidence']*100:.2f}% (for **{results['class']}**)")
                
                raw_predictions = results['raw_predictions']
                
                # Combine class names and scores and sort for visualization
                class_scores = list(zip(CABBAGE_CLASS_NAMES, raw_predictions))
                class_scores.sort(key=lambda x: x[1], reverse=True)
                
                top_n = 5
                chart_data = {
                    'Class': [score[0] for score in class_scores[:top_n]], 
                    'Confidence': [f"{score[1]*100:.2f}%" for score in class_scores[:top_n]]
                }
                
                st.dataframe(chart_data, use_container_width=True)


# ==============================================================================
# 4. SIDEBAR AND FOOTER (Optional/Navigation)
# ==============================================================================

# --- Sidebar Content ---
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <h3>Cabbage Detector Settings</h3>
    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown(f"**Current Coverage:** {', '.join([c.replace('cabbage ', '') for c in CABBAGE_CLASS_NAMES])}")
st.sidebar.markdown(f"**Minimum Confidence (Threshold):** {REJECTION_THRESHOLD*100:.0f}%")
st.sidebar.markdown(f"**Model Input Size:** {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
st.sidebar.markdown("---")
st.sidebar.markdown("For assistance, upload a clear image or use the camera to focus on the symptoms.")


# --- Hiding default Streamlit pages/menu ---
hide_pages_css = """
<style>
/* Hide the default Streamlit page selector (sidebar pages menu) */
[data-testid="stSidebarNav"] ul {
    display: none !important;
}
</style>
"""
st.markdown(hide_pages_css, unsafe_allow_html=True)


# --- Placeholder for Navigation/Full Model Scope (Removed unnecessary logic) ---
# The complex HTML rendering for the full list of plants and the full class names array 
# are typically handled by a main dashboard file or simplified for a single-page app.
# Keeping the sidebar navigation to 'Cabbage' status for this specific app file.