import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Import MobileNetV2-specific preprocessing logic (Normalization to [-1, 1])
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import numpy as np
import io
from PIL import Image
import os 
import pathlib
# Import the custom intervention logic 
from skumawiki_intervention import get_skumawiki_interventions 

# --- 1. CONFIGURATION ---

# Base directory setup
BASE_DIR = pathlib.Path(__file__).parent 

# Construct the full path to the model (Placeholder - update this)
MODEL_PATH = os.path.join(BASE_DIR, 'APP_MODELS', 'SKUMAWIKI_mobileNet_model2.h5') 
REJECTION_THRESHOLD = 0.50 # 50% confidence minimum
IMG_SIZE = (248, 248) # Model input size (as defined in your template)
TITLE = "AgroVision AI: Skumawiki Leaf Detector"

# Define the specialized list of class names for this specific model
# NOTE: Ensure this list exactly matches the labels used during your model training!
CLASS_NAMES = [
    'General Healthy Leaf',
    'General Powdery Mildew',
    'General Bacterial Blight'
]
NUM_CLASSES = len(CLASS_NAMES)

# --- 2. STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title=TITLE, layout="centered")

# --- CSS INJECTION (Using skumawiki_style.css) ---
def inject_custom_css(file_path):
    """Reads a local CSS file and injects it into the Streamlit app."""
    try:
        abs_path = os.path.join(BASE_DIR, file_path)
        with open(abs_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception:
        pass 

inject_custom_css("style.css") 

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_trained_model(path):
    """Loads the model from the .h5 file."""
    try:
        # Load the model
        model = load_model(path)
        return model
    except Exception as e:
        # Fallback for local testing or missing file
        st.error(f"Error loading model: Could not load model at '{path}'. Error: {e}")
        st.warning("Using a placeholder model structure (initialize with random weights). Please ensure your model file is present.")
        
        # Define a minimal architecture for error handling/placeholder
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        
        base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights=None)
        x = base_model.output
        x = GlobalAveragePooling2D()(x) 
        outputs = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        return model

model = load_trained_model(MODEL_PATH)

# --- 4. PREDICTION FUNCTION ---
def preprocess_and_predict(img_data, model, class_names, img_size):
    """
    Preprocesses the image data, applies MobileNetV2 normalization, and predicts.
    """
    if model is None:
        return {"status": "error", "diagnosis": "Model Not Available", "message": "The AI model failed to load."}

    try:
        # Handle Streamlit uploaded file object
        if hasattr(img_data, 'getvalue'): 
            img = Image.open(io.BytesIO(img_data.getvalue())).convert('RGB')
        else:
            img = Image.open(img_data).convert('RGB')

        # Resize and convert
        img = img.resize(img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply MobileNetV2 preprocessing: Normalization to [-1, 1]
        img_array = mobilenet_preprocess(img_array)

        # Make prediction
        predictions = model.predict(img_array)[0]
        
        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index]
        predicted_class = class_names[predicted_index]
        
        
        # --- HEALTH PROMPT AND REJECTION LOGIC ---
        
        # 1. Check for Low Confidence/Rejection
        if confidence < REJECTION_THRESHOLD:
            return {
                "status": "inconclusive", 
                "diagnosis": "Diagnosis Inconclusive (Low Confidence)", 
                "confidence": confidence,
                "class": predicted_class,
                "message": (
                    f"The model's top prediction is too low ({confidence*100:.2f}% < {REJECTION_THRESHOLD*100:.0f}%). "
                    "Please try again with a clearer image focusing on the symptoms."
                ),
                "raw_predictions": predictions
            }
        
        # 2. Check for Healthy Leaf
        elif 'healthy' in predicted_class.lower():
            return {
                "status": "healthy",
                "diagnosis": "Plant Appears Healthy!",
                "confidence": confidence,
                "class": predicted_class,
                "message": (
                    f"The leaf appears **healthy and vibrant** with {confidence*100:.2f}% confidence. "
                    "Continue proper care and monitoring."
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
                "message": f"Potential issue identified: **{predicted_class}**. See recommended actions below.",
                "raw_predictions": predictions
            }

    except Exception as e:
        st.exception(e)
        return {"status": "error", "diagnosis": "Processing Error", "message": f"An unexpected error occurred: {e}"}


# --- 5. STREAMLIT APP INTERFACE ---

st.markdown(
    f"""
    <div class="big-font">{TITLE}</div>
    <div class="subheader-font">Targeting Powdery Mildew and Bacterial Blight</div>
    """, 
    unsafe_allow_html=True
)
st.markdown("This application uses a MobileNetV2-based model for rapid diagnosis of two common plant afflictions.")

st.info(f"The model detects {NUM_CLASSES} classes: " + ', '.join([c.replace('General ', '') for c in CLASS_NAMES]))

# Simplified Input Section (Camera and Uploader)
st.markdown("### 📸 Image Input")
st.warning("Please upload a clear photo of the suspected diseased leaf for accurate diagnosis.")

col_cam, col_upload = st.columns(2)

with col_cam:
    camera_img = st.camera_input("1. Capture Photo of the Leaf")

with col_upload:
    uploaded_file = st.file_uploader("2. Upload Image from Device", type=["jpg", "jpeg", "png"])

# 6. EXECUTION AND RESULTS DISPLAY
input_data = None
if camera_img is not None:
    input_data = camera_img
elif uploaded_file is not None:
    input_data = uploaded_file

if input_data is not None:
    st.markdown("---")
    st.subheader("Image Selected for Analysis")
    
    image_col, result_col = st.columns([1, 1])

    with image_col:
        st.image(input_data, caption='Ready for analysis.', use_column_width=True)
    
    with result_col:
        if st.button('Diagnose with Skumawiki AI', key='diagnose_button', use_container_width=True):
            
            with st.spinner('Running specialized plant analysis...'):
                results = preprocess_and_predict(input_data, model, CLASS_NAMES, IMG_SIZE)
            
            st.markdown("### 🔬 Diagnosis Result")
            
            # --- Displaying Results based on Status ---
            
            if results["status"] == "healthy":
                st.success(f"**Status:** {results['class']}")
                st.markdown(f"**Confidence:** {results['confidence']*100:.2f}%")
                st.balloons()
                st.info(results['message'])
                
            elif results["status"] == "warning":
                st.error(f"**Disease Detected:** {results['class']}")
                st.markdown(f"**Confidence:** {results['confidence']*100:.2f}%")
                
                # Intervention Box
                disease_name = results['class']
                intervention_data = get_skumawiki_interventions(disease_name) 
                
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="intervention-title">
                            ⚠️ Recommended Action for: {disease_name.replace('General ', '')}
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.markdown(f"**Treatment Summary:** {intervention_data['title']}")
                    st.markdown("**Action Plan:**")
                    for i, action in enumerate(intervention_data['action']):
                        st.markdown(f"**{i+1}.** {action}")

            elif results["status"] == "inconclusive":
                st.warning(f"**Status:** {results['diagnosis']}")
                st.markdown(f"**Confidence:** {results['confidence']*100:.2f}%")
                st.info(results['message'])

            elif results["status"] == "error":
                st.exception(results['message'])

            # --- Display Top Predictions ---
            if results["status"] != "error":
                st.markdown("---")
                st.markdown("### 📊 Top Prediction Scores")
                
                raw_predictions = results['raw_predictions']
                class_scores = list(zip(CLASS_NAMES, raw_predictions))
                class_scores.sort(key=lambda x: x[1], reverse=True)
                
                top_classes = [score[0].replace('General ', '') for score in class_scores[:NUM_CLASSES]]
                top_confidences = [score[1] for score in class_scores[:NUM_CLASSES]]

                chart_data = {
                    'Class': top_classes, 
                    'Confidence': [f"{c*100:.2f}%" for c in top_confidences]
                }
                
                st.dataframe(chart_data, use_container_width=True)

# --- 7. SIDEBAR INSTRUCTIONS ---
st.sidebar.header("Model Information")
st.sidebar.markdown(f"**Target Diseases:** Powdery Mildew, Bacterial Blight")
st.sidebar.markdown(f"**Minimum Confidence (Threshold):** {REJECTION_THRESHOLD*100:.0f}%")
st.sidebar.markdown(f"**Model Type:** MobileNetV2 (Transfer Learning)")
st.sidebar.markdown(f"**Input Size:** {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
st.sidebar.markdown("---")
st.sidebar.markdown("For best results, capture the leaf with visible symptoms in good lighting.")