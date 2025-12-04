import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os 
import pathlib
from intervention import get_interventions

# --- 1. CONFIGURATION ---




BASE_DIR = pathlib.Path(__file__).parent 

# 2. Construct the full path: Joins the base directory, the folder name ('MODELS'), 
# and the file name to create an absolute path.
MODEL_PATH = os.path.join(BASE_DIR, 'MODELS', 'mobileNet_model2.h5')
REJECTION_THRESHOLD = 0.50


IMG_SIZE = (248, 248)  # Model input size
TITLE = "AgroVision AI : Crop Disease Detector"

# --- 2. STREAMLIT PAGE CONFIG (MOVED TO THE TOP) ---
# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title=TITLE, layout="wide")


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
# We will use this to guide the user on which section their plant belongs to.

VEGETABLE_CLASSES = [
    'Corn',
    'Potatoe',
    'Tomato',
    'Pepper',
    'soybean',
    'skumawiki',
    'onion',
    'Cabbage'

]
FRUIT_CLASSES = [
    'Apple', 
    'Grape', 
    'Cherry',
    'Strawberry',
    'Raspberry',
    'Peach',
    'Orange'
]
    




# --- CSS INJECTION FUNCTION ---
def inject_custom_css(file_path):
    """Reads a local CSS file and injects it into the Streamlit app."""
    try:
        with open(file_path) as f:
            # st.markdown injects the CSS wrapped in <style> tags
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at path: {file_path}")
    except Exception as e:
        st.error(f"Error injecting CSS: {e}")

# Call the function with the correct path to apply styles immediately
inject_custom_css("style.css")

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
        st.error("Invalid image input type")
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
    <div class="subheader-font">Real Time Crop Disease Diagnosis</div>
    """, 
    unsafe_allow_html=True
)

# Create the two main tabs
tab_vegetables, tab_fruits = st.tabs(["🥕 Vegetable Crops", "🍎 Fruit Crops"])

# Helper function to place content within tabs
def render_tab_content(crop_type, class_list):
    st.markdown(
        f"### 📸 <div class='text-white' style='display:inline;'>{crop_type} Image Input</div>",
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
            
            # Implement rejection mechanism based on confidence threshold
            if confidence < REJECTION_THRESHOLD:
                predicted_class = "Uncertain Prediction - Please try again with a clearer image."
        
        #  Display Results
        
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


            st.markdown(
                f"""
                <div class="intervention-box">
                    <div class="intervention-title">
                        <i class="fa-solid fa-stethoscope"></i> Suggested Action
                    </div>
                """, unsafe_allow_html=True
            )
            
           # 🛑 INTERVENTION LOGIC: PULLING DATA FROM EXTERNAL FILE
            intervention_data = get_interventions(predicted_class)
            
            st.markdown(f"#### {intervention_data['title']}")
            
            # Display actions in a structured, numbered list
            for i, action in enumerate(intervention_data['action']):
                st.markdown(f"**{i+1}.** {action}")
            # 🛑 END INTERVENTION LOGIC
        
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

# The content to be wrapped
st.sidebar.markdown(
    """
    div class="sidebar1">
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
            <li>Select the appropriate tab for your crop type (Vegetable or Fruit).</li>
            <li>Use the camera input to take a photo of the leaf or upload an image from your device.</li>
            <li>Click the 'Diagnose Leaf' button to analyze the image.</li>
            <li>Review the diagnosis results and suggested interventions.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)

