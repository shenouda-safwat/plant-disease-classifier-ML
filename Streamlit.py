import streamlit as st
import pickle
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from PIL import Image

image = Image.open("assets/planet.jpg")
st.image(image, caption="Plant Disease Classifier", use_column_width=True)

with open('rf_plant_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

def extract_color_histogram(img, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(gray,
               orientations=9,
               pixels_per_cell=(16, 16),
               cells_per_block=(2, 2),
               block_norm='L2-Hys',
               visualize=False,
               feature_vector=True)

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    (hist_lbp, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 10),
                                 range=(0, 9))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-6)

    color_hist = extract_color_histogram(img)

    return np.concatenate([hog_feat, hist_lbp, color_hist])

class_names = ['Healthy', 'Powdery', 'Rust']

st.title("Plant Disease Detection 🌿")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)

    img_resized = cv2.resize(img, (224, 224))
    features = extract_features(img_resized).reshape(1, -1)

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    best_idx = np.argmax(proba)
    best_class = class_names[best_idx]

    st.markdown(
        f"""
        <div style="
            background-color: #e0f7fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        ">
            <h2 style="color: #00796b; margin-bottom: 5px;">Prediction Result</h2>
            <h1 style="font-size: 48px; margin: 0;">
                <span style="color: #2e7d32;">{best_class}</span>
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
