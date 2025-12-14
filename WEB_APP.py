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
import pandas as pd 

# Assuming 'intervention.py' exists in the same directory
# NOTE: Ensure intervention.py handles the new per-plant classes
from intervention import get_interventions

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

# Base directory setup
BASE_DIR = pathlib.Path(__file__).parent 

# Paths and Constants
SINGLE_MODEL_PATH = os.path.join(BASE_DIR, "leaf_disease_mobilenet_final2.h5")
REJECTION_THRESHOLD = 0.50
# NOTE: IMG_SIZE is now determined dynamically by the model mapping below
TITLE = "AgroVision AI : Crop Disease Detector"

# Background Image Setup: Ensure these files are in the same directory
BACKGROUND_IMAGE_PATH = os.path.join(BASE_DIR, 'vege2.jpeg')
CSS_PATH = os.path.join(BASE_DIR, 'style.css') 
CSS_PLACEHOLDER = "BACKGROUND_IMAGE_PLACEHOLDER" 

# --- Categorization for Sidebar ---
VEGETABLE_CLASSES = ['Corn', 'Potato', 'Tomato', 'Pepper Bell', 'Soybean', 'Onion', 'Cabbage']
FRUIT_CLASSES = ['Apple', 'Grape', 'Cherry', 'Strawberry', 'Raspberry', 'Peach', 'Orange']
ALL_PLANTS = VEGETABLE_CLASSES + FRUIT_CLASSES

# ==============================================================================
# 1.1.  DYNAMIC MODEL, CLASS, AND SIZE MAPPINGS (THE CORE FIX)
# ==============================================================================


DYNAMIC_MODEL_MAPPING = {
    "Potato": {"path": os.path.join(BASE_DIR, "APP_MODELS/_model.h5"), "img_size": (224, 224)}, # Default
    "Tomato": {"path": os.path.join(BASE_DIR, "APP_MODELS/TOMATO_mobileNet_model.h5"), "img_size": (128,128)},
    "Apple": {"path": os.path.join(BASE_DIR, "APP_MODELS/APPLE_mobileNet_model.h5"), "img_size": (224, 224)},
    
    "Corn": {"path": os.path.join(BASE_DIR, "APP_MODELS/CORN_mobileNet_model7.h5"), "img_size": (248, 248)}, 
    
    "Grape": {"path": os.path.join(BASE_DIR, "APP_MODELS/GRAPES_mobileNet_model7.h5"), "img_size": (248,248)}, 
    
    "Cherry": {"path": os.path.join(BASE_DIR, "APP_MODELS/CHERRY_mobileNet_model7.h5"), "img_size": (248,248)},
    "Strawberry": {"path": os.path.join(BASE_DIR, "APP_MODELS/STRAWBERRY_mobileNet_model.h5"), "img_size": (224, 224)},
    "Raspberry": {"path": os.path.join(BASE_DIR, "APP_MODELS/RASPBERRY_mobileNet_model.h5"), "img_size": (224, 224)},
    "Peach": {"path": os.path.join(BASE_DIR, "APP_MODELS/peach_model.h5"), "img_size": (224, 224)},
    "Orange": {"path": os.path.join(BASE_DIR, "APP_MODELS/ORANGES_mobileNet_model2.h5"), "img_size": (128,128)},
    "Pepper Bell": {"path": os.path.join(BASE_DIR, "APP_MODELS/PEPPER_mobileNet_model7.h5"), "img_size": (224, 224)},
    "Soybean": {"path": os.path.join(BASE_DIR, "APP_MODELS/soybean_model.h5"), "img_size": (224, 224)},
    "Onion": {"path": os.path.join(BASE_DIR, "APP_MODELS/ONION_mobileNet_model.h5"), "img_size": (224, 224)},
    "Cabbage": {"path": os.path.join(BASE_DIR, "APP_MODELS/CABBAGE_mobileNet_model7.h5"), "img_size": (248,248)},
}


# ⚠️ IMPORTANT: DEFINE THE SPECIFIC CLASS NAMES FOR EACH MODEL (Remains the same as your input)
CLASS_NAMES_MAPPING = {
    "Potato": ['Potato Early Blight', 'Potato Late Blight', 'Potato Healthy'],
    "Tomato": ['Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Healthy', 'Tomato late blight',
               'Tomato leaf mold', 'Tomato septoria leaf spot', 'Tomato spider mites Two-spotted spider mite',
               'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus'],
    "Apple": ['Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Apple Healthy'],
    "Corn": ['Corn Common Rust', 'Corn Northern leaf blight', 'Corn Cercospora Leaf Spot gray leaf spot', 'Corn Healthy'],
    "Grape": ['Grape Black Rot', 'Grape Esca (Black Measles)', 'Grape Leaf Blight (Isariopsis Leaf Spot)', 'Grape Healthy'],
    "Cherry": ['Cherry Powdery Mildew', 'Cherry Healthy'],
    "Strawberry": ['Strawberry Leaf Scorch', 'Strawberry Healthy'],
    "Raspberry": ['raspberry healthy', 'raspberry leaf spot'],
    "Peach": ['peach healthy', 'peach bacterial spot', 'peach leaf curl', 'peach powdery mildew', 'peach leaf disease'],
    "Orange": ['orange citrus greening', 'orange leaf curl', 'orange leaf disease', 'orange leaf spot', 'Orange Healthy'],
    "Pepper Bell": ['Pepper Bell Bacterial Spot', 'Pepper Bell Healthy'],
    "Soybean": ['soybean healthy', 'soybean frog eye leaf spot', 'soybean rust', 'soybean powdery mildew'],
    "Onion": ['onion downy mildew', 'onion healthy leaf', 'onion leaf blight', 'onion purple blotch','onion thrips damage'],
    "Cabbage": ['cabbage black rot','cabbage healthy','cabbage clubroot','cabbage downy mildew','cabbage leaf disease'],
}

# We no longer use FULL_CLASS_NAMES for prediction, but keep it for reference or potential UI elements
FULL_CLASS_NAMES_REFERENCE = [
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
    
# NEW FUNCTION: Reset the entire analysis flow
def reset_app():
    st.session_state.selected_plant = None
    st.session_state.analysis_run = False
    st.session_state.prediction_result = None
    st.rerun() # Trigger a rerun to go back to the welcome state
    
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
# 3. UTILITY FUNCTIONS (CSS Injection remains unchanged)
# ==============================================================================

# ---------------------------------------------------
# 3.1. DYNAMIC MODEL LOADING FUNCTION (MODIFIED)
# ---------------------------------------------------
@st.cache_resource
def load_specific_model(plant_name):
    """
    Dynamically loads the specific Keras model based on the selected plant name.
    """
    # Check the NEW DYNAMIC_MODEL_MAPPING structure
    if plant_name not in DYNAMIC_MODEL_MAPPING:
        st.error(f"Configuration Error: No model file path found for '{plant_name}'.")
        return "DummyModel"
    
    # Retrieve the model path from the new dict structure
    model_path = DYNAMIC_MODEL_MAPPING[plant_name]["path"]
    
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at path: {model_path}. Using Dummy Model.")
        return "DummyModel" 
    
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        return "DummyModel" # Return the dummy model string on failure

# ---------------------------------------------------
# 3.2. CSS Injection execution 
# ---------------------------------------------------
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
    img_base64_css = f"""
    /* 1. REMOVE background from main content area (stVerticalBlock) */
    [data-testid="stVerticalBlock"] > div:nth-child(1) {{
        background-image: none ;
        background-color: transparent ;
        padding: 0 ;
        margin-bottom: 0 ;
        color: inherit ;
    }}
    
    /* 2. APPLY background image to the entire sidebar (top to bottom) */
    [data-testid="stSidebar"] > div:first-child {{
        background-image: 
            linear-gradient(
                rgba(20, 70, 30, 0.8),
                rgba(85, 60, 30, 0.7)
            ),
            url("{base64_url}"); /* Use the encoded image */
        background-size: cover;
        background-attachment: fixed; 
        background-position: center;
    }}
    
    /* Ensure the main title container looks correct now that the main background is gone */
    .title-container {{
        background: transparent;
    }}
    """
    
    try:
        with open(file_path) as f:
            css_content = f.read()
            # If the original style.css contains a placeholder, replace it. 
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
# 5. MODEL LOADING STATUS
# ==============================================================================
st.success("✅ Machine Learning Infrastructure Initialized.") 


# ==============================================================================
# 6. PREDICTION FUNCTION (MODIFIED TO USE DYNAMIC IMG_SIZE)
# ==============================================================================
def preprocess_and_predict(img_data, model, class_names, img_size):
    """
    Preprocesses the image data and returns the prediction or a mock prediction 
    if the model is a DummyModel.
    'img_size' is the dynamic target size (e.g., (224, 224) or (124, 124)).
    """
    
    # --- MOCK MODEL LOGIC ---
    if model == "DummyModel":
        # (Mock logic remains the same)
        time.sleep(1)
        
        selected_plant = st.session_state.selected_plant if st.session_state.selected_plant else "Tomato"
        plant_prefix = selected_plant.split(' ')[0].lower()
        
        relevant_classes = [c for c in class_names if c.lower().startswith(plant_prefix)]
        
        if not relevant_classes:
            return "No Relevant Class Found", 0.0, np.zeros(len(class_names))

        healthy_options = [c for c in relevant_classes if 'healthy' in c.lower()]
        diseased_options = [c for c in relevant_classes if 'healthy' not in c.lower()]
        
        if np.random.rand() < 0.6 and diseased_options:
            predicted_class = np.random.choice(diseased_options)
        elif healthy_options:
             predicted_class = np.random.choice(healthy_options)
        else:
            predicted_class = np.random.choice(relevant_classes)
            
        confidence = np.random.uniform(0.75, 0.95)
            
        raw_predictions = np.zeros(len(class_names))
        try:
            pred_index = class_names.index(predicted_class)
            raw_predictions[pred_index] = confidence
        except ValueError:
            pass 
        return predicted_class, confidence, raw_predictions

    # --- REAL MODEL PROCESSING ---
    try:
        # Image loading logic remains
        if isinstance(img_data, io.BytesIO):
            img = Image.open(img_data).convert('RGB')
        elif isinstance(img_data, Image.Image):
            img = img_data
        elif hasattr(img_data, 'getvalue'): # For Streamlit uploaded file object
             img = Image.open(io.BytesIO(img_data.getvalue())).convert('RGB')
        else:
            img = Image.open(img_data).convert('RGB')

        # CRITICAL FIX: Use the dynamically provided img_size for resizing
        img = img.resize(img_size) 
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 

        # Prediction uses the specific Keras model object
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index]
        predicted_class = class_names[predicted_index] 

        return predicted_class, confidence, predictions
    except Exception as e:
        st.error(f"Prediction failed: Exception encountered. Error: {e}")
        return "Prediction Error", 0.0, np.zeros(len(class_names))


# ==============================================================================
# 7. STREAMLIT APP INTERFACE (MAIN CONTENT)
# ==============================================================================


st.markdown(
    f"""
    <div class="title-container1">
        <div class="big-font">{TITLE}</div>
        <div class="subheader-font">Real Time Crop Disease Diagnosis</div>
    </div>
    """, 
    unsafe_allow_html=True
)


# ==============================================================================
# 8. INPUT AND ANALYSIS SECTION (MODIFIED TRIGGER)
# ==============================================================================

if st.session_state.selected_plant:
    selected_plant = st.session_state.selected_plant
    st.markdown("---") 
    
    
    # ----------------------------------------------------------------------
    with st.container(border=True): 
        st.markdown(
            f"""<div class="diagnosis"> <h3>📸 Input for {selected_plant} Leaf Diagnosis</h3></div>""",
            unsafe_allow_html=True
        )
        
        st.info(f"Please use one of the two options below to submit an image of the **{selected_plant}** leaf.")
        
        col_cam, col_upload = st.columns(2)
        
        # 1. Camera Input
        camera_img = col_cam.camera_input(
            label="Capture Photo", 
            key="camera_input"
        )
        
        # 2. Upload File
        uploaded_file = col_upload.file_uploader(
            f"2. Upload File from Device (Desktop/Local Files)", 
            type=["jpg", "jpeg", "png"], 
            key="uploader_input"
        )
    # ----------------------------------------------------------------------
    
    # --- Logic: Determine Input and Execute Prediction ---
    input_data = None
    if camera_img is not None:
        input_data = camera_img # Streamlit object directly
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
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Prediction button
            if st.button(f'Diagnose {selected_plant} Leaf Now', key='diagnose_button', use_container_width=True):
                st.session_state.analysis_run = True 
                
                # --- NEW DYNAMIC MODEL LOGIC HERE ---
                current_model = load_specific_model(selected_plant)
                
                # 1. NEW STEP: Retrieve the required image size from the DYNAMIC_MODEL_MAPPING
                # Uses (224, 224) as a safe fallback if the configuration is missing
                required_img_size = DYNAMIC_MODEL_MAPPING.get(selected_plant, {}).get("img_size", (224, 224))
                
                # 2. Get the specific class names for the selected plant
                current_class_names = CLASS_NAMES_MAPPING.get(selected_plant, [])
                
                if not current_class_names:
                    st.error(f"Configuration Error: Class names not found for {selected_plant}.")
                else:
                    with st.spinner(f'Running analysis for {selected_plant} leaf with specialized model (Size: {required_img_size})...'):
                        predicted_class, confidence, raw_predictions = preprocess_and_predict(
                            input_data, 
                            current_model, 
                            current_class_names, 
                            required_img_size # <--- PASS THE DYNAMIC SIZE
                        )

                    st.session_state.prediction_result = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "raw_predictions": raw_predictions,
                        "class_names_used": current_class_names # Save the class names used
                    }
    
    # --- Display Results if analysis_run is True and results are available ---
    if st.session_state.analysis_run and st.session_state.prediction_result:
        results = st.session_state.prediction_result
        
        # --- Diagnosis/Rejection Logic ---
        predicted_class = results['predicted_class']
        confidence = results['confidence']

        if confidence < REJECTION_THRESHOLD:
            final_diagnosis = "Uncertain Prediction - Please try again with a clearer image."
        elif predicted_class == "Prediction Error" or predicted_class == "No Relevant Class Found":
            final_diagnosis = "Prediction Error - Model could not process image."
        else:
            # The cross-validation warning is less necessary now, but kept for robustness
            plant_prefix = selected_plant.lower().split(' ')[0]
            if not predicted_class.lower().startswith(plant_prefix):
                 st.warning(f"⚠️ **Model Confusion:** The prediction '{predicted_class}' does not match the selected crop. Review the scores below.")
                 
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
            st.sidebar.markdown("---")

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
                <h3>📊 Prediction Scores (Scoped to {selected_plant})</h3>
            </div>""", unsafe_allow_html=True)
            
            # --- Display Scores using the specific class names ---
            raw_predictions = results['raw_predictions']
            class_names_used = results['class_names_used']
            
            if raw_predictions is not None and np.any(raw_predictions):
                class_scores = list(zip(class_names_used, raw_predictions))
                class_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Display all results in a data frame
                scores_df = pd.DataFrame(class_scores, columns=['Class', 'Probability'])
                scores_df['Probability'] = (scores_df['Probability'] * 100).round(2).astype(str) + '%'
                
                st.dataframe(scores_df, use_container_width=True, hide_index=True)
            else:
                st.info("No detailed prediction scores are available due to an error or uncertainty.")

            # --- REFRESHER BUTTON ---
            st.button(
                label="🚀 Start New Analysis / Choose New Plant",
                key="new_analysis_button",
                on_click=reset_app, 
                help="Click here to clear the current results and select a new plant.",
                type="secondary",
                use_container_width=True
            )

# --- Initial Message if no plant is selected (The New Home Page) ---
else:
    st.markdown(
    """
    <div class="custom-info-box">
        👈 <strong>Select a crop</strong> from the sidebar to begin the leaf disease diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)

    # 1. Marketing Banner (Top Text)
    st.markdown(
        f"""
        <div class="home-marketing-box">
            <h2>Protect Your Crops with Intelligent Disease Detection</h2>
            <p style='font-size:1.2em; text-align:center;'>
                Upload a leaf image and get instant AI diagnosis for 60+ crop diseases. 
                Learn treatment strategies and protect your harvest.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 2. Metric Icons (using columns)
    col_diseases, col_plants, col_accuracy = st.columns(3)
    
    total_classes = len(FULL_CLASS_NAMES_REFERENCE)
    total_plants = len(ALL_PLANTS)
    
    with col_diseases:
        st.markdown(f"""
            <div class='metric-item'>
                <p style='font-size: 2.5em; margin: 0;'>🦠</p>
                <h4>{total_classes}+</h4>
                <p>Disease Classes</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_plants:
        st.markdown(f"""
            <div class='metric-item'>
                <p style='font-size: 2.5em; margin: 0;'>🌿</p>
                <h4>{total_plants}</h4>
                <p>Crop Types Covered</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_accuracy:
        st.markdown(f"""
            <div class='metric-item'>
                <p style='font-size: 2.5em; margin: 0;'>🎯</p>
                <h4>90%+</h4>
                <p>Prediction Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 3. Footer Markdown 
    st.markdown(
    """
    <div class="footer-container">
        <div class="footer-title">Protecting Your Harvest</div>
        <div class="footer-text">
            Understanding plant diseases is the first step to healthier crops and better yields.<br>
            Our AI-powered platform helps farmers identify and treat diseases early.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


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
st.sidebar.markdown("---")

st.sidebar.button(
    label="New Analysis / Home",
    key="sidebar_home_button",
    on_click=reset_app,
    help="Click to go back to the main app interface and clear all selections.",
    type="primary",
    use_container_width=True
)

st.sidebar.markdown(
    """
    <div class="sidebar2">
        <h3>SELECT CROP FOR DIAGNOSIS</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# 1. Prepare the options list
options_list = ["--- Select a Crop ---"] + ALL_PLANTS

# Determine the index of the currently selected plant
if st.session_state.selected_plant in ALL_PLANTS:
    default_index = options_list.index(st.session_state.selected_plant)
else:
    default_index = 0

# 2. Create the Dropdown Selector
selected_option = st.sidebar.selectbox(
    label="Choose a crop from the list below:",
    options=options_list,
    index=default_index,
    key="plant_selector_dropdown",
    label_visibility="collapsed"
)

# 3. Update the session state based on the selection
if selected_option and selected_option != "--- Select a Crop ---":
    if st.session_state.selected_plant != selected_option:
        set_plant(selected_option)

st.sidebar.markdown("---")