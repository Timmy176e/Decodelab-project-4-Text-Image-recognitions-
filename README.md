# Image & Text Recognition — Project 4

Industrial training project (DecodeLabs, Batch 2026) demonstrating two
machine perception pipelines built on pre-trained models.

## What it does

- **Text Recognition (OCR):** Preprocesses an image (grayscale → Gaussian
  blur → adaptive thresholding → deskew) and extracts text using Tesseract
  OCR, with a confidence score per word.
- **Object Detection:** Runs a pre-trained MobileNet-SSD model (OpenCV DNN,
  trained on PASCAL VOC) to detect and label objects with bounding boxes
  and per-detection confidence scores.

Both modes enforce a minimum confidence threshold (default 80%, adjustable)
before a result is accepted and displayed.

## Tech stack

- Python, OpenCV (`cv2.dnn`), Tesseract OCR (`pytesseract`)
- MobileNet-SSD (Caffe model, PASCAL VOC classes)
- Streamlit (web interface)

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires the Tesseract binary installed separately on your system
(e.g. `sudo apt install tesseract-ocr` on Linux).

## Usage

1. Choose a mode in the sidebar: Text Recognition or Object Detection.
2. Upload an image.
3. Adjust the confidence threshold if needed.
4. View the extracted text or labeled bounding boxes.

## Live demo

[Streamlit app link — add after deployment]
