import streamlit as st
import cv2
import numpy as np
from io import BytesIO

def convert_image_to_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray_image
    blur_image = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    inv_blur = 255 - blur_image
    sketch = cv2.divide(gray_image, inv_blur, scale=255.0)
    return sketch

def main():
    st.title("Pencil Sketch Converter")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption="Uploaded Image")
        
        sketch = convert_image_to_sketch(image)
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        
        # Encode the image as a JPEG in memory
        _, buffer = cv2.imencode('.jpg', sketch_rgb)
        byte_io = BytesIO(buffer)
        
        st.image(sketch_rgb, caption="Pencil Sketch")
        
        # Provide a download button with the JPEG byte buffer
        st.download_button(
            label="Download Sketch",
            data=byte_io,
            file_name="sketch.jpg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()