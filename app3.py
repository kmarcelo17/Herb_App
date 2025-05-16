import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# Remedies Dictionary (sample — add more as needed)
remedies = {
    "Peppermint": "Soothes digestion, reduces nausea, relieves headaches.",
    "Basil": "Anti-inflammatory, antibacterial, good for stress and digestion.",
    "Turmeric": "Treats inflammation, pain, digestive issues, arthritis, depression.",
    # Add all other herbs and remedies...
}

# Load pre-trained model
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

model = load_model()

# Preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Compute embedding
def compute_embedding(img):
    processed = preprocess_image(img)
    return model.predict(processed)[0]

# Cosine similarity
def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

# Load dataset images
def load_dataset_images(folder):
    image_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                image_paths.append(os.path.join(root, f))
    return image_paths

# Find similar images
def find_similar_images(ref_img, dataset_folder, top_k=5):
    ref_emb = compute_embedding(ref_img)
    dataset_paths = load_dataset_images(dataset_folder)
    matches = []

    for path in dataset_paths:
        try:
            img = Image.open(path).convert("RGB")
            emb = compute_embedding(img)
            score = cosine_similarity(ref_emb, emb)
            matches.append((path, score))
        except Exception as e:
            print(f"Skipped {path}: {e}")

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]

# ---------------- Streamlit UI ----------------

st.title("🌿 Herbal Plant Remedy Detector")
st.write("Upload a plant image and select a dataset folder to detect similar herbal plants and learn about their remedies.")

# Upload reference image
uploaded_file = st.file_uploader("Upload a reference plant image", type=["jpg", "jpeg", "png"])
dataset_folder = st.text_input("Enter path to your herbal plant dataset folder")

if uploaded_file and dataset_folder:
    try:
        ref_img = Image.open(uploaded_file).convert("RGB")
        st.image(ref_img, caption="Uploaded Reference Image", width=300)

        if st.button("Find Similar Herbs"):
            with st.spinner("Processing..."):
                matches = find_similar_images(ref_img, dataset_folder)

            for i, (path, score) in enumerate(matches, 1):
                herb_name = os.path.basename(os.path.dirname(path))
                remedy = remedies.get(herb_name, "No known remedy available.")
                match_img = Image.open(path)

                st.markdown(f"### {i}. {herb_name}")
                st.image(match_img, width=200)
                st.markdown(f"**Similarity Score:** `{score:.4f}`")
                st.markdown(f"**Remedy:** {remedy}")
                st.markdown("---")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Please upload a reference image and enter the dataset folder path.")

