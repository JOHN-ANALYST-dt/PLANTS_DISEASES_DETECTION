import base64
import os
import streamlit as st

def encode_image_to_base64(path):
    """
    Reads a local image, encodes it to Base64, and prepends the MIME type
    to create a complete Data URL string for web use.
    
    This function handles .png and .jpg extensions.
    """
    try:
        # 1. Determine MIME type based on file extension
        ext = os.path.splitext(path)[1].lower()
        if ext == '.png':
            mime_type = "image/png"
        elif ext in ('.jpg', '.jpeg'):
            mime_type = "image/jpeg"
        else:
            print(f"Warning: Unsupported image type {ext}. Returning 'none'.")
            return "none"
        
        # 2. Open and encode the file
        with open(path, "rb") as f:
            data = f.read()
            encoded_string = base64.b64encode(data).decode('utf-8')
            
        # 3. Construct the Data URL
        return f"data:{mime_type};base64,{encoded_string}"
        
    except FileNotFoundError:
        st.error(f"Background image '{path}' not found. Using solid background.")
        return "none"
    except Exception as e:
        st.error(f"Error during image encoding: {e}")
        return "none"

# 🛑 3.1. Perform Encoding 
# The encoder is now called with the correct path to veget.PNG
img_base64_url = encode_image_to_base64(BACKGROUND_IMAGE_PATH)


def inject_custom_css(file_path, base64_url):
    """
    Reads a local CSS file and injects it into the Streamlit app, 
    replacing the placeholder with the Base64 URL.
    """
    try:
        with open(file_path) as f:
            css_content = f.read()
            
            # Replace the placeholder in CSS with the actual Base64 URL
            final_css = css_content.replace(CSS_PLACEHOLDER, base64_url)
            
            # Inject the final CSS
            st.markdown(f'<style>{final_css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at path: {file_path}")
    except Exception as e:
        st.error(f"Error injecting CSS: {e}")

# 🛑 3.2. Inject Styles Immediately After Page Config
inject_custom_css("style.css", img_base64_url)