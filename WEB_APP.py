import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os 
import pathlib
import base64
import time

# Assuming 'intervention.py' exists in the same directory
# NOTE: Ensure this file is available in your environment.
from intervention import get_interventions

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

# Base directory setup
BASE_DIR = pathlib.Path(__file__).parent 

# Paths and Constants
MODEL_PATH = os.path.join(BASE_DIR, "inceptionv3_model2.h5")
REJECTION_THRESHOLD = 0.50
IMG_SIZE = (124, 124) 

TITLE = "AgroVision AI : Crop Disease Detector"

# Background Image Setup: Ensure these files are in the same directory
BACKGROUND_IMAGE_PATH = os.path.join(BASE_DIR, 'vege2.jpeg')
CSS_PATH = os.path.join(BASE_DIR, 'style.css') # Assuming style.css is required
CSS_PLACEHOLDER = "BACKGROUND_IMAGE_PLACEHOLDER" # Placeholder in style.css

# Define the full list of class names (MUST match training order)
FULL_CLASS_NAMES = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 
    'cabbage black rot','cabbage healthy','cabbage clubroot','cabbage downy mildew','cabbage leaf disease',
    'Corn Common Rust', 'Corn Northern leaf blight', 'Corn Cercospora Leaf Spot gray leaf spot',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
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
    'peach healthy', 'peach bacterial spot', 'peach leaf curl', 'peach powdery mildew', 'peach leaf disease',
    'orange citrus greening', 'orange leaf curl', 'orange leaf disease', 'orange leaf spot',
    'onion downy mildew', 'onion healthy leaf', 'onion leaf blight', 'onion purple blotch','onion thrips damage'
]

# Function to update the selected plant in session state
def set_plant(plant_name):
    st.session_state.selected_plant = plant_name
    st.session_state.analysis_run = False # Reset analysis when a new plant is selected
    st.session_state.prediction_result = None
    
# NEW FUNCTION: Reset the entire analysis flow
def reset_app():
    st.session_state.selected_plant = None
    st.session_state.analysis_run = False
    st.session_state.prediction_result = None
    st.rerun() # Trigger a rerun to go back to the welcome state


# --- Categorization for Sidebar ---
VEGETABLE_CLASSES = ['Corn', 'Potato', 'Tomato', 'Pepper Bell', 'Soybean', 'Onion', 'Cabbage']
FRUIT_CLASSES = ['Apple', 'Grape', 'Cherry', 'Strawberry', 'Raspberry', 'Peach', 'Orange']
ALL_PLANTS = VEGETABLE_CLASSES + FRUIT_CLASSES


# ==============================================================================
# 2. APP SETUP (MUST BE FIRST EXECUTABLE COMMANDS)
# ==============================================================================

# --- 2.1. STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title=TITLE, layout="wide")

# --- 2.2. SESSION STATE INITIALIZATION (CRITICAL FOR NAVIGATION) ---
if 'selected_plant' not in st.session_state:
    st.session_state.selected_plant = None
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
    
# Function to update the selected plant in session state
def set_plant(plant_name):
    st.session_state.selected_plant = plant_name
    st.session_state.analysis_run = False # Reset analysis when a new plant is selected
    st.session_state.prediction_result = None
    
# --- 2.3. HIDE DEFAULT STREAMLIT PAGE SELECTOR ---
hide_pages_css = """
<style>
/* Hide the default Streamlit page selector (sidebar pages menu) */
[data-testid="stSidebarNav"] ul {
    display: none !important;
}
</style>
"""
st.markdown(hide_pages_css, unsafe_allow_html=True)


# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================

def encode_image_to_base64(path):
    """Reads a local image and encodes it to a Base64 Data URL string."""
    if not os.path.exists(path):
        st.error(f"Background image file not found at expected path: {path}. Using solid background.")
        return "none"
        
    try:
        ext = os.path.splitext(path)[1].lower()
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
    # Consolidated background image injection logic 
    img_base64_css = f"""
    [data-testid="stVerticalBlock"] > div:nth-child(1) {{
        background-image: 
            linear-gradient(
                rgba(20, 70, 30, 0.8),
                rgba(85, 60, 30, 0.7)
            ),
            url("{base64_url}"); /* Use the encoded image */
        background-size: cover;
        background-position: center;
        padding: 40px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }}
    """
    
    try:
        with open(file_path) as f:
            css_content = f.read()
            # If the original style.css contains a placeholder, replace it. 
            # Otherwise, just append the background image CSS.
            if CSS_PLACEHOLDER in css_content:
                final_css = css_content.replace(CSS_PLACEHOLDER, base64_url) + img_base64_css
            else:
                 final_css = css_content + img_base64_css
                 
        st.markdown(f'<style>{final_css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at path: {file_path}. Applying dynamic background only.")
        st.markdown(f'<style>{img_base64_css}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error injecting CSS: {e}")


# --- 4. BACKGROUND IMAGE & CSS INJECTION ---
img_base64_url = encode_image_to_base64(BACKGROUND_IMAGE_PATH)
inject_custom_css(CSS_PATH, img_base64_url)


# ==============================================================================
# 4.2. CUSTOM CSS for Diagnose Button Styling (FINAL, AGGRESSIVE STYLING)
# ==============================================================================
st.markdown("""
<style>
/* FIX: Use a highly specific selector and distinct color (Blue) 
    to ensure the style overrides all defaults.
*/
div[data-testid*="diagnose_button"] button {
    background-color: #007bff !important; /* Strong Blue for Action */
    color: white !important; 
    font-size: 1.3em !important; /* Very large text */
    padding: 18px 35px !important; /* Large padding */
    border-radius: 15px !important; /* Rounded corners */
    border: 4px solid #0056b3 !important; /* Thicker, darker border */
    font-weight: 900 !important; /* Extra bold */
    letter-spacing: 1px; /* Spacing out the text */
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5) !important; /* Strong shadow */
    transition: all 0.2s ease;
}

div[data-testid*="diagnose_button"] button:hover {
    background-color: #0056b3 !important; /* Darker Blue on hover */
    border-color: #004085 !important;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 5. LOAD MODEL
# ==============================================================================
@st.cache_resource
def load_trained_model(path):
    """Loads the model from the .h5 file or simulates a load."""
    time.sleep(1) # Simulate loading time
    
    # Check if the model file actually exists
    if not os.path.exists(path):
        st.error(f"Model file not found at path: {path}")
        st.warning("Using **mock model** for demonstration.")
        return "DummyModel"
    
    try:
        model = load_model(path)
        st.success("✅ Machine Learning Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Using **mock model** for demonstration.")
        return "DummyModel"

model = load_trained_model(MODEL_PATH)


# ==============================================================================
# 6. PREDICTION FUNCTION
# ==============================================================================
def preprocess_and_predict(img_data, model, class_names, img_size):
    """
    Preprocesses the image data and returns the prediction or a mock prediction 
    if the model is a DummyModel.
    """
    if model == "DummyModel":
        time.sleep(1)
        
        # Mock prediction logic: Use selected plant for a more relevant mock result
        selected_plant = st.session_state.selected_plant if st.session_state.selected_plant else "Tomato"
        plant_prefix = selected_plant.split(' ')[0].lower()
        
        # Filter for relevant mock classes
        relevant_classes = [c for c in class_names if c.lower().startswith(plant_prefix)]
        
        if not relevant_classes:
            return "No Relevant Class Found", 0.0, np.zeros(len(class_names))

        # Randomly choose a healthy or diseased class from the relevant list
        if np.random.rand() < 0.6:
            predicted_class = np.random.choice([c for c in relevant_classes if 'healthy' not in c.lower()])
        else:
            predicted_class = np.random.choice([c for c in relevant_classes if 'healthy' in c.lower()] or relevant_classes)
            
        confidence = np.random.uniform(0.75, 0.95)
            
        raw_predictions = np.zeros(len(class_names))
        try:
            pred_index = class_names.index(predicted_class)
            raw_predictions[pred_index] = confidence
        except ValueError:
            pass 
        return predicted_class, confidence, raw_predictions

    # Real model processing (if model were available)
    try:
        # Image loading logic remains
        if isinstance(img_data, io.BytesIO):
            img = Image.open(img_data).convert('RGB')
        elif isinstance(img_data, Image.Image):
            img = img_data
        elif hasattr(img_data, 'getvalue'): # For Streamlit uploaded file object
            img = Image.open(img_data).convert('RGB')
        else:
             raise ValueError("Invalid image input type")

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
        # Ensures a NumPy array is returned for unpacking
        return "Prediction Error", 0.0, np.zeros(len(class_names))


# ==============================================================================
# 7. STREAMLIT APP INTERFACE (MAIN CONTENT)
# ==============================================================================


st.markdown(
        f"""
        <div class="title-container">
            <div class="big-font">{TITLE}</div>
            <div class="subheader-font">Real Time Crop Disease Diagnosis</div>
        </div>
        """, 
        unsafe_allow_html=True
    )


# ==============================================================================
# 8. INPUT AND ANALYSIS SECTION
# ==============================================================================

if st.session_state.selected_plant:
    # This assignment runs only when a plant is selected
    selected_plant = st.session_state.selected_plant
    st.markdown("---") 
    
    
    # This entire block MUST be indented under the 'if' condition
    # ----------------------------------------------------------------------
    with st.container(border=True): 
        st.markdown(
            f"""<div class="diagnosis"> <h3>📸 Input for {selected_plant} Leaf Diagnosis</h3></div>""",
            unsafe_allow_html=True
        )
        
        st.info(f"Please use one of the two options below to submit an image of the **{selected_plant}** leaf.")
        
        col_cam, col_upload = st.columns(2)
        
        # 1. Camera Input: Using a custom key and label for CSS targeting
        camera_img = col_cam.camera_input(
            label="Capture Photo", # Simplified label
            key="camera_input"
        )
        
        # 2. Upload File
        uploaded_file = col_upload.file_uploader(
            f"2. Upload File from Device (Desktop/Local Files)", 
            type=["jpg", "jpeg", "png"], 
            key="uploader_input"
        )
    # ----------------------------------------------------------------------
    # END: New Input Box
    # ----------------------------------------------------------------------
    
    # --- Logic: Determine Input and Execute Prediction ---
    input_data = None
    if camera_img is not None:
        input_data = Image.open(camera_img)
    elif uploaded_file is not None:
        input_data = uploaded_file
        
    # Only show analysis button if an image is provided
    if input_data is not None:
        st.markdown("---")
        st.markdown("""<div class="analysis">
                        <h3>Image Selected for Analysis</h3>
                    </div>""", unsafe_allow_html=True)
        
        image_col, result_col = st.columns([1, 1])

        with image_col:
            st.image(input_data, caption=f'{selected_plant} Leaf Ready for Analysis.', use_column_width=True)
            
        with result_col:
            # Prediction button
            if st.button(f'Diagnose {selected_plant} Leaf Now', key='diagnose_button', use_container_width=True):
                st.session_state.analysis_run = True 
                
                with st.spinner(f'Running analysis for {selected_plant} leaf...'):
                    # The unpacking line is here. The fix ensures 3 values are always returned.
                    predicted_class, confidence, raw_predictions = preprocess_and_predict(
                        input_data, model, FULL_CLASS_NAMES, IMG_SIZE
                    )

                st.session_state.prediction_result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "raw_predictions": raw_predictions
                }
                
                # Removed st.rerun() in the previous step to fix the warning.
    
    # --- Display Results if analysis_run is True and results are available ---
    if st.session_state.analysis_run and st.session_state.prediction_result:
        results = st.session_state.prediction_result
        
        # --- Diagnosis/Rejection Logic ---
        predicted_class = results['predicted_class']
        confidence = results['confidence']

        if confidence < REJECTION_THRESHOLD:
            final_diagnosis = "Uncertain Prediction - Please try again with a clearer image."
        elif predicted_class == "Prediction Error":
            final_diagnosis = "Prediction Error - Model could not process image."
        else:
            # Check if the prediction matches the selected plant
            plant_prefix = selected_plant.lower().split(' ')[0]
            if not predicted_class.lower().startswith(plant_prefix):
                 st.warning(f"⚠️ The model detected an issue primarily found in other crops (e.g., '{predicted_class}'). Showing results for **{selected_plant}**.")
                 
            final_diagnosis = predicted_class
        
        # --- Display Results UI ---
        st.markdown(f""" 
        <div class="analysis">
            <h3>🔍 Analysis Results for {selected_plant}</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2]) 
        
        with col1:
            st.markdown(f"""
            <div class="diagnosis"> 
                <h3>🔬 Diagnosis Result</h3>
            </div>
            """,unsafe_allow_html=True)

            if 'Healthy' in final_diagnosis:
                st.success(f"**Status:** {final_diagnosis}")
                st.balloons()
            elif 'Uncertain' in final_diagnosis or 'Error' in final_diagnosis:
                st.warning(f"**Status:** {final_diagnosis}")
            else:
                st.error(f"**Disease Detected:** {final_diagnosis}")
            
            st.info(f"**Confidence:** {confidence*100:.2f}%")

            # INTERVENTION LOGIC
            intervention_data = get_interventions(final_diagnosis)
            st.markdown(
                f"""
                <div class="intervention-box">
                    <div class="intervention-title">
                        Suggested Action
                    </div>
                    #### {intervention_data['title']}
                """, unsafe_allow_html=True
            )
            
            for i, action in enumerate(intervention_data['action']):
                st.markdown(f"**{i+1}.** {action}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="confid">
                <h3>📊 Top Prediction Scores (Scoped to {selected_plant})</h3>
            </div>""", unsafe_allow_html=True)
            
            # --- CRITICAL CHANGE: Filter Scores by Selected Plant ---
            raw_predictions = results['raw_predictions']
            
            # Only proceed with sorting/display if raw_predictions is a valid numpy array (not None or 0s from error)
            if raw_predictions is not None and np.any(raw_predictions):
                class_scores = list(zip(FULL_CLASS_NAMES, raw_predictions))
                plant_prefix = selected_plant.lower().split(' ')[0] 
                
                # Filter scores to only include classes relevant to the selected plant (e.g., 'cabbage black rot' for 'Cabbage')
                filtered_scores = [
                    (c, s) for c, s in class_scores if c.lower().startswith(plant_prefix)
                ]
                
                filtered_scores.sort(key=lambda x: x[1], reverse=True)
                
                top_n = 5
                top_classes = [score[0] for score in filtered_scores[:top_n]]
                top_confidences = [score[1] for score in filtered_scores[:top_n]]

                chart_data = {
                    'Class': top_classes, 
                    'Confidence': [f"{c*100:.2f}%" for c in top_confidences]
                }
                st.dataframe(chart_data, use_container_width=True)
            else:
                 st.info("No detailed prediction scores are available due to an error or uncertainty.")

            # --- INSERT THE NEW REFRESHER BUTTON CODE HERE ---
            st.button(
                label="🚀 Start New Analysis / Choose New Plant",
                key="new_analysis_button",
                on_click=reset_app, 
                help="Click here to clear the current results and select a new plant.",
                type="secondary",
                use_container_width=True
            )

# --- Initial Message if no plant is selected ---
else:
    st.markdown("---")
    st.info("👈 **Select a crop** from the sidebar to begin the leaf disease diagnosis. The image input section will appear here.")
# ==============================================================================
# 9. SIDEBAR INSTRUCTIONS & NAVIGATION
# ==============================================================================



# --- Sidebar Content ---
st.sidebar.markdown(
    """
    <div class="sidebar1">
        <h3>Current Model Coverage</h3>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.button(
    label="New Analysis / Home",
    key="sidebar_home_button",
    on_click=reset_app,
    help="Click to go back to the main app interface and clear all selections.",
    type="primary", # Makes the button stand out
    use_container_width=True
)


# ---------- Sidebar navigation (Buttons to update state) ----------
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar2">
        <h3>SELECT PLANT FOR PREDICTION</h3>
    </div>
    """,
    unsafe_allow_html=True
)

for plant in ALL_PLANTS:
    # Use st.button to trigger the set_plant function via on_click
    is_selected = st.session_state.selected_plant == plant
    
    st.sidebar.button(
        label=plant,
        key=f"plant_btn_{plant}",
        on_click=set_plant,
        args=(plant,),
        # Highlight the selected button
        type="primary" if is_selected else "secondary", 
        use_container_width=True
    )
    
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Trained on:** {len(FULL_CLASS_NAMES)} Disease Classes")
st.sidebar.markdown(
    """
    <div class="sidebar2">
        <h3>How to Use This App</h3>
        <ol>
            <li>**Select a crop** from the list above.</li>
            <li>The image input area will appear below.</li>
            <li>Upload or take a photo of the leaf.</li>
            <li>Click the 'Diagnose Leaf' button.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)