# gui/dashboard.py
import streamlit as st
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.preprocessor import Preprocessor
from modules.segmentor import segmentor
from modules.identifier import Identifier
from modules.feature_extractor import FeatureExtractor
from modules.grader import Grader

st.set_page_config(
    page_title="Fruit Ripeness Assessment System",
    page_icon="🍌",
    layout="wide"
)

# --- Sidebar: settings only ---
with st.sidebar:
    st.header("Settings")
    fruit_options  = list(config.FRUIT_PROFILES.keys())
    selected_fruit = st.selectbox("Select Fruit", fruit_options)
    config.ACTIVE_FRUIT = selected_fruit
    st.divider()
    uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg", "jpeg", "png"])

# --- Main panel ---
st.title("🍌 Fruit Ripeness Assessment System")
st.markdown("Upload a fruit image from the sidebar to assess its ripeness level.")

if uploaded_file is None:
    st.info("Awaiting image upload...")
else:
    # Decode image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run pipeline
    preprocessed = Preprocessor().preprocess(img)
    s = segmentor()
    mask = s.create_mask(preprocessed)
    contours = s.get_contours(mask)

    if len(contours) == 0:
        st.error("No fruit detected. Please try another image.")
    else:
        fruit, avg_hue, circular = Identifier().identify(img, mask, contours[0])
        features = FeatureExtractor().extract(preprocessed, mask)
        result = Grader().grade(features)

        # Draw contour
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

        # Convert to RGB for Streamlit
        img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        contour_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)

        # Results metrics
        st.subheader("Assessment Results")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Fruit",fruit)
        r2.metric("Ripeness",result)
        r3.metric("Hue Mean",f"{features['hue_mean']:.2f}")
        r4.metric("Sat Mean",f"{features['sat_mean']:.2f}")
        r5.metric("Blemish Ratio", f"{features['blemish_ratio']:.4f}")

        st.divider()

        # Image panels
        st.subheader("Image Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Original")
            st.image(img_rgb, use_container_width=True)
        with col2:
            st.caption("Mask")
            st.image(mask, use_container_width=True)
        with col3:
            st.caption("Contour")
            st.image(contour_rgb, use_container_width=True)