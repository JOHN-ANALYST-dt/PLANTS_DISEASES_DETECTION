import streamlit as st
import numpy as np
import io
from PIL import Image
import os 
import pathlib
import base64
import time

# Mock dependencies as the actual TensorFlow model and crop image are not provided
try:
    from intervention import get_interventions
except ImportError:
    # Define a mock intervention function if the file doesn't exist yet
    def get_interventions(disease_name):
        if 'Healthy' in disease_name:
            return {'title': 'Keep Up the Good Work!', 'action': ['Continue current irrigation and fertilization schedule.', 'Monitor regularly for early signs of pests.']}
        elif 'Bacterial Spot' in disease_name:
            return {'title': 'Management for Bacterial Spot', 'action': ['Remove and destroy infected leaves or plants immediately.', 'Apply copper-based bactericides every 7-10 days.', 'Avoid overhead watering to keep foliage dry.']}
        else:
            return {'title': 'General Crop Health Tip', 'action': ['Consult a local agricultural extension office for a specific diagnosis.', 'Ensure proper soil drainage and ventilation.']}

# --- 1. CONFIGURATION ---

BASE_DIR = pathlib.Path(__file__).parent 

# Paths and Constants
MODEL_PATH = os.path.join(BASE_DIR, 'PEPPER_mobileNet_model7.h5')
REJECTION_THRESHOLD = 0.50 
IMG_SIZE = (248, 248) 

TITLE = "Pepper Vision AI: Bell Pepper Disease Detector"

# Mock Background Image Path (must exist for base64 encoding to work)
BACKGROUND_IMAGE_PATH = './pepper_bg.jpeg' 
CSS_PLACEHOLDER = "BACKGROUND_IMAGE_PLACEHOLDER" 


# --- 2. STREAMLIT PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title=TITLE, layout="wide")


# --- 3. UTILITY FUNCTIONS (Inlined for simplicity) ---

def encode_image_to_base64(path):
    """Reads a local image and encodes it to a Base64 Data URL string."""
    # Since we don't have the image, we'll use a placeholder URL 
    # and warn the user. In a real environment, the image must exist.
    st.warning("Using placeholder background image URL. Please ensure 'pepper_bg.jpeg' and 'style.css' are available.")
    # Using a placeholder image URL for robust deployment
    return "https://placehold.co/1200x800/228B22/FFFFFF/png?text=Pepper+Background"


def inject_custom_css(file_path, base64_url):
    """
    Reads local CSS, replaces the placeholder with the Base64 URL, 
    and injects the final styles into the Streamlit app.
    """
    # Simplified CSS injection assuming style.css might not be present, 
    # but still respecting the placeholder pattern
    try:
        with open(file_path) as f:
            css_content = f.read()
            final_css = css_content.replace(CSS_PLACEHOLDER, f"url('{base64_url}')")
            st.markdown(f'<style>{final_css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback inline CSS for critical elements
        st.markdown(f"""
            <style>
                .stApp {{
                    background-color: #f0f2f6;
                }}
                .title-container {{
                    background-color: #38761d;
                    padding: 20px;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .big-font {{
                    font-size: 40px;
                    font-weight: bold;
                }}
                .subheader-font {{
                    font-size: 20px;
                }}
            </style>
        """, unsafe_allow_html=True)


# --- 4. BACKGROUND IMAGE INJECTION ---

# 4.1. Perform Encoding using the new path
img_base64_url = encode_image_to_base64(BACKGROUND_IMAGE_PATH)

# 4.2. Inject Styles Immediately After Page Config
inject_custom_css("style.css", img_base64_url)


# Define the specific class names for Pepper (MUST match training order)
FULL_CLASS_NAMES = [
    'Pepper Bell Bacterial Spot', 
    'Pepper Bell Healthy'
]

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
        # Mock prediction logic focusing on Pepper diseases
        if np.random.rand() < 0.7:
            # 70% chance of a healthy diagnosis
            predicted_class = "Pepper Bell Healthy"
            confidence = np.random.uniform(0.80, 0.98)
        else:
            # 30% chance of Bacterial Spot
            predicted_class = "Pepper Bell Bacterial Spot"
            confidence = np.random.uniform(0.70, 0.95)
            
        raw_predictions = np.zeros(len(class_names))
        try:
            pred_index = class_names.index(predicted_class)
            raw_predictions[pred_index] = confidence
            # Add a small mock confidence for the other class
            other_index = 1 - pred_index 
            raw_predictions[other_index] = 1 - confidence # Mock remainder
        except ValueError:
            pass 
        return predicted_class, confidence, raw_predictions

    # Real model processing (place actual TensorFlow/Keras prediction code here)
    try:
        # ... real image preprocessing and prediction logic ...
        pass
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Prediction Error", 0.0, None

    # Fallback for real model logic if it fails
    return "DummyModel Failure", 0.0, np.zeros(len(class_names))


# --- 7. STREAMLIT APP INTERFACE ---

st.markdown(
    f"""
    <div class="title-container">
        <div class="big-font">{TITLE}</div>
        <div class="subheader-font">Focused Real-Time Bell Pepper Disease Diagnosis</div>
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown("## 📸 Bell Pepper Leaf Image Input")
st.markdown(f"**This detector covers:** Bell Pepper Bacterial Spot and Bell Pepper Healthy.")
st.warning("For mobile use, please select the 'Take a Photo' option provided by the camera input.")

# Use the Streamlit Camera Input as the primary method
camera_img = st.camera_input("Take a Photo (Bell Pepper Leaf)", key="camera_pepper")

st.markdown("---")
st.write("### ⬆️ Or Upload a File")
uploaded_file = st.file_uploader("Choose an image from your device (Bell Pepper Leaf)...", type=["jpg", "jpeg", "png"], key="uploader_pepper")


# 8. CONSOLIDATED INPUT CHECK
input_data = None
if camera_img is not None:
    input_data = Image.open(camera_img)
elif uploaded_file is not None:
    input_data = uploaded_file


if input_data is not None:
    st.write("")
    
    # Display the selected image 
    with st.container():
        st.markdown(f"""
        <div class="selector">
            <h3>Image Selected for Diagnosis</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(input_data, caption='Ready for analysis.', use_column_width=True)
    
    # Prediction button
    if st.button('Diagnose Bell Pepper Leaf', key='diagnose_button'):
        with st.spinner('Analyzing image for disease...'):
            predicted_class, confidence, raw_predictions = preprocess_and_predict(
                input_data, model, FULL_CLASS_NAMES, IMG_SIZE
            )
            
            # Implement rejection mechanism
            if confidence < REJECTION_THRESHOLD:
                predicted_class = "Uncertain Prediction - Please try again with a clearer image."
        
        # Display Results
        st.markdown(f""" 
        <div class="analysis">
            <h3>🔍 Analysis Results</h3>
        </div>
        """,
        unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2]) 

        with col1:
            st.markdown(f"""
            <div class="diagnosis"> 
                <h3>🔬 Diagnosis Result</h3>
            </div>
            """,unsafe_allow_html=True
            )
            
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
                        Suggested Action
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
            st.markdown(f"""
            <div class="confid">
                <h3>📊 Confidence Scores (Top 2)</h3>
            </div>""", unsafe_allow_html=True)
            
            class_scores = list(zip(FULL_CLASS_NAMES, raw_predictions))
            class_scores.sort(key=lambda x: x[1], reverse=True)
            
            top_n = len(FULL_CLASS_NAMES) # Only 2 classes, display both
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
    <div class="sidebar1" style='padding: 15px; background-color: #d9ead3; border-radius: 8px;'>
        <h3 style='color: #38761d;'>Bell Pepper Model Coverage</h3>
    </div>
    <div style='padding: 10px;'>
        <p>This application is trained to detect:</p>
        <ul>
            <li>Bell Pepper Bacterial Spot</li>
            <li>Bell Pepper Healthy</li>
        </ul>
        <hr>
        <h4 style='color: #4a86e8;'>How to Use:</h4>
        <ol>
            <li>Use the camera or upload a clear, focused image of a pepper leaf.</li>
            <li>Click 'Diagnose Bell Pepper Leaf'.</li>
            <li>Review the diagnosis and suggested actions.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)