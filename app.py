import os
import urllib.request

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

# ---------- MobileNet-SSD setup (only needed for Object Detection mode) ----------
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
CAFFEMODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_PATH = "MobileNetSSD_deploy.caffemodel"

@st.cache_resource
def load_model():
    if not os.path.exists(PROTOTXT_PATH):
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
    if not os.path.exists(CAFFEMODEL_PATH):
        urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL_PATH)
    return cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

# ---------- OCR pipeline ----------
def run_ocr(image_np, conf_threshold):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=10
    )

    data = pytesseract.image_to_data(adaptive, output_type=pytesseract.Output.DICT)

    boxed = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    accepted_words = []

    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        conf = int(data['conf'][i])
        if word and conf >= conf_threshold:
            accepted_words.append((word, conf))
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return adaptive, boxed, accepted_words

# ---------- Object detection pipeline ----------
def run_detection(image_np, net, conf_threshold):
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    (h, w) = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(image_bgr, scalefactor=0.007843, size=(300, 300), mean=127.5)
    net.setInput(blob)
    detections = net.forward()

    boxed = image_bgr.copy()
    accepted = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence * 100 >= conf_threshold:
            class_id = int(detections[0, 0, i, 1])
            label_name = CLASSES[class_id]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            accepted.append((label_name, confidence, (x1, y1, x2, y2)))

            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{label_name}: {confidence * 100:.1f}%"
            cv2.putText(boxed, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return boxed, accepted

# ---------- Main app ----------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Uploaded image", use_container_width=True)

    if mode == "Text Recognition (OCR)":
        if st.button("Run OCR"):
            with st.spinner("Processing..."):
                adaptive, boxed, accepted_words = run_ocr(image_np, confidence_threshold)

            st.image(adaptive, caption="Preprocessed (adaptive threshold)", use_container_width=True, clamp=True)
            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB),
                      caption=f"Accepted words (confidence ≥ {confidence_threshold}%)",
                      use_container_width=True)

            st.subheader("Extracted text")
            if accepted_words:
                st.write(" ".join(w for w, c in accepted_words))
                with st.expander("Per-word confidence"):
                    for word, conf in accepted_words:
                        st.write(f"**{word}** — {conf}%")
            else:
                st.warning("No words met the confidence threshold.")

    else:  # Object Detection
        if st.button("Run Object Detection"):
            with st.spinner("Loading model and running detection..."):
                net = load_model()
                boxed, accepted = run_detection(image_np, net, confidence_threshold)

            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB),
                      caption=f"Detections (confidence ≥ {confidence_threshold}%)",
                      use_container_width=True)

            st.subheader("Detected objects")
            if accepted:
                for label_name, confidence, box in accepted:
                    st.write(f"**{label_name}** — {confidence*100:.1f}%  `box={box}`")
            else:
                st.warning("No objects met the confidence threshold.")
else:
    st.info("Upload an image to get started.")
