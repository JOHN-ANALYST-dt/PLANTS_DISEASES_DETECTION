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
# Assuming 'intervention.py' and 'style.css' exist in the same directory
from intervention import get_interventions 

# --- 1. CONFIGURATION ---

# Base directory setup
BASE_DIR = pathlib.Path(__file__).parent 

# Construct the full path to the model
MODEL_PATH = os.path.join(BASE_DIR, 'MODELS', 'TOMATO_mobileNet_model.h5')
REJECTION_THRESHOLD = 0.50 # 50% confidence minimum
IMG_SIZE = (248,248) # Model input size
TITLE = "AgroVision AI: Tomato Leaf Detector"

# --- 2. STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title=TITLE, layout="centered")

# Define the specialized list of class names for TOMATO
# Ensure these class names exactly match the labels used during your model training!
TOMATO_CLASS_NAMES = [
    'tomato healthy leaf',
    'tomato bacterial spot',
    'tomato early blight',
    'tomato late blight',
    'tomato leaf mold',
    'tomato mosaic virus',
    'tomato yellow leaf curl virus'
]

# --- CSS INJECTION (Assuming style.css exists) ---
def inject_custom_css(file_path):
    """Reads a local CSS file and injects it into the Streamlit app."""
    try:
        # Check if the file path is relative and adjust if necessary
        abs_path = os.path.join(BASE_DIR, file_path)
        with open(abs_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at path: {file_path}. Using default Streamlit styling.")
    except Exception as e:
        st.error(f"Error injecting CSS: {e}")

inject_custom_css("style.css") # Assuming style.css is provided

# --- 3. LOAD MODEL ---
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

model = load_trained_model(MODEL_PATH)

# --- 4. PREDICTION FUNCTION ---
def preprocess_and_predict(img_data, model, class_names, img_size):
    """
    Preprocesses the image data and returns the prediction result dictionary.
    Includes MobileNetV2-specific preprocessing.
    """
    if model is None:
        return {"status": "error", "diagnosis": "Model Not Available", "message": "The AI model failed to load."}

    try:
        # Handle different input types (file_uploader, camera_input)
        if isinstance(img_data, io.BytesIO):
            img = Image.open(img_data).convert('RGB')
        elif isinstance(img_data, Image.Image):
            img = img_data
        elif hasattr(img_data, 'getvalue'): # Handle Streamlit uploaded file object
            img = Image.open(io.BytesIO(img_data.getvalue())).convert('RGB')
        else:
            raise ValueError("Invalid image input type")

        # Resize to the model's expected input size
        img = img.resize(img_size)
        
        # Convert PIL Image to Numpy array
        img_array = image.img_to_array(img)
        
        # Expand dimensions to match the input shape (1, H, W, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply MobileNetV2-specific preprocessing: Normalization to [-1, 1]
        img_array = mobilenet_preprocess(img_array)

        # Make prediction
        predictions = model.predict(img_array)[0]
        
        # Get the predicted class index and confidence
        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index]
        predicted_class = class_names[predicted_index]
        
        
        # --- HEALTH PROMPT AND REJECTION LOGIC ---
        
        # 1. Check for Low Confidence/Rejection
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
        
        # 2. Check for Healthy Leaf (MANDATORY REQUEST)
        elif 'healthy' in predicted_class.lower():
            return {
                "status": "healthy",
                "diagnosis": "Excellent News! Healthy Tomato Leaf Detected",
                "confidence": confidence,
                "class": predicted_class,
                "message": (
                    f"Your tomato leaf appears **vibrant and free of disease!** "
                    f"Confidence: {confidence*100:.2f}%. Ensure adequate water and nutrient supply, "
                    "and stake your plants to prevent ground contact."
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
        st.exception(e)
        return {"status": "error", "diagnosis": "Processing Error", "message": f"An unexpected error occurred: {e}"}


# --- 5. STREAMLIT APP INTERFACE ---

st.markdown(
    f"""
    <style>
    .big-font {{
        font-size:36px !important;
        font-weight: 800;
        color: #DC143C; /* Crimson - a deep red for tomatoes */
    }}
    .subheader-font {{
        font-size:24px !important;
        color: #228B22; /* Forest Green for freshness */
        margin-bottom: 20px;
    }}
    </style>
    <div class="big-font">{TITLE}</div>
    <div class="subheader-font">Specialized Diagnosis for Tomato Crops</div>
    """, 
    unsafe_allow_html=True
)

st.info("This application is specialized for detecting the following **Tomato** issues: " + ', '.join(TOMATO_CLASS_NAMES))

# Simplified Input Section (Camera and Uploader)
st.markdown("### 📸 Image Input")
st.warning("For best results, take a clear photo of the *affected area* of a single leaf, or upload a high-quality image.")

col_cam, col_upload = st.columns(2)

with col_cam:
    # Camera Input (for mobile button functionality)
    camera_img = st.camera_input("1. Take a Photo of the Leaf")

with col_upload:
    # File Uploader Input
    uploaded_file = st.file_uploader("2. Upload an Image from Device", type=["jpg", "jpeg", "png"])

# 6. EXECUTION AND RESULTS DISPLAY
input_data = None
# Determine which input was used
if camera_img is not None:
    # Convert camera input to PIL Image immediately for consistent handling
    input_data = Image.open(camera_img) 
elif uploaded_file is not None:
    input_data = uploaded_file

# Check if an image is provided
if input_data is not None:
    st.markdown("---")
    st.subheader("Image Selected for Analysis")
    
    # Use a container to display image and prediction side-by-side or stacked
    image_col, result_col = st.columns([1, 1])

    with image_col:
        st.image(input_data, caption='Ready for analysis.', use_column_width=True)
    
    with result_col:
        # Prediction button
        if st.button('Diagnose Leaf Now', key='diagnose_button', use_container_width=True):
            
            with st.spinner('Running specialized Tomato leaf analysis...'):
                results = preprocess_and_predict(input_data, model, TOMATO_CLASS_NAMES, IMG_SIZE)
            
            st.markdown("### 🔬 Diagnosis Result")
            
            # --- Displaying Results based on Status ---
            
            if results["status"] == "healthy":
                st.success(f"**Status:** {results['class']}")
                st.balloons()
                st.markdown(f"**Confidence:** {results['confidence']*100:.2f}%")
                st.markdown(
                    f"""
                    <div class="healthy-prompt">
                        <div class="emoji">🍅🌿</div>
                        <p class="message">{results['message']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
                
            elif results["status"] == "warning":
                st.error(f"**Disease Detected:** {results['class']}")
                st.markdown(f"**Confidence:** {results['confidence']*100:.2f}%")
                
                # Intervention Box
                intervention_data = get_interventions(results['class']) # Use the full class name
                st.markdown(
                    f"""
                    <div class="intervention-box">
                        <div class="intervention-title">
                            ⚠️ Immediate Action Required
                        </div>
                        <p class="mb-2">{results['message']}</p>
                        <h4 class='text-lg font-bold text-red-700 mt-3'>Recommended Treatment:</h4>
                        <p>{intervention_data['title']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                # Display actions in a structured, numbered list
                for i, action in enumerate(intervention_data['action']):
                    st.markdown(f"**{i+1}.** {action}")

            elif results["status"] == "inconclusive":
                st.warning(f"**Status:** {results['diagnosis']}")
                st.markdown(f"**Confidence:** {results['confidence']*100:.2f}%")
                st.info(results['message'])

            elif results["status"] == "error":
                st.exception(results['message'])

            # --- Display Top Predictions (for all non-error results) ---
            if results["status"] != "error":
                st.markdown("---")
                st.markdown("### 📊 Top Prediction Scores")
                
                raw_predictions = results['raw_predictions']
                
                # Combine class names and scores and sort for visualization
                class_scores = list(zip(TOMATO_CLASS_NAMES, raw_predictions))
                class_scores.sort(key=lambda x: x[1], reverse=True)
                
                top_n = 5
                top_classes = [score[0] for score in class_scores[:top_n]]
                top_confidences = [score[1] for score in class_scores[:top_n]]

                chart_data = {
                    'Class': top_classes, 
                    'Confidence': [f"{c*100:.2f}%" for c in top_confidences]
                }
                
                st.dataframe(chart_data, use_container_width=True)

# --- 7. SIDEBAR INSTRUCTIONS ---
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <h3>Tomato Detection Status</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(f"**Current Coverage:** {', '.join([c.replace('tomato ', '') for c in TOMATO_CLASS_NAMES])}")
st.sidebar.markdown(f"**Minimum Confidence (Threshold):** {REJECTION_THRESHOLD*100:.0f}%")
st.sidebar.markdown(f"**Model Input Size:** {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
st.sidebar.markdown("---")
st.sidebar.markdown("For assistance, upload a clear image or use the camera to focus on the symptoms.")