import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

st.set_page_config(page_title="Project 4: Image & Text Recognition", layout="centered")
st.title("Image & Text Recognition")
st.caption("DecodeLabs Project 4 — OCR + Object Detection pipeline")

mode = st.sidebar.radio("Choose a mode:", ["Text Recognition (OCR)", "Object Detection"])
confidence_threshold = st.sidebar.slider("Confidence threshold (%)", 50, 99, 80)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Uploaded image", use_container_width=True)
    st.write(f"Mode selected: **{mode}**")
else:
    st.info("Upload an image to get started.")
