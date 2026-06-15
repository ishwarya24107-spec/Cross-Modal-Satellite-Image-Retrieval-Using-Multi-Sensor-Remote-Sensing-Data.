import streamlit as st
import numpy as np
import os
import tifffile as tffi
from extractor_backend import SatelliteFeatureExtractor

# --- CONFIGURATION ---
# Point this to the original s2 folder where the app can retrieve the full images to show the user
S2_FOLDER = r"C:\Users\ishwa\Downloads\archive\s2_validation\s2_validation"

st.set_page_config(layout="wide")
st.title("🛰️ Satellite Cross-Modal Image Retrieval System")
st.write("Upload a raw **Sentinel-1 SAR Radar patch** to search the database and find its matching **Sentinel-2 Optical RGB patch**.")

# Load the vector database we built in the last step
@st.cache_resource
def load_database():
    vectors = np.load("gallery_vectors.npy")
    filenames = np.load("gallery_names.npy")
    extractor = SatelliteFeatureExtractor()
    return vectors, filenames, extractor

try:
    db_vectors, db_filenames, extractor = load_database()
    st.success(f"Database loaded successfully! {len(db_filenames)} optical assets indexed.")
except Exception as e:
    st.error("Could not load database files. Did you run index_gallery.py first?")
    st.stop()

# --- FILE PROCESSOR (Same normalization you verified!) ---
def process_tif(file_obj, is_sar=True):
    img = tffi.imread(file_obj)
    if img.ndim == 3 and img.shape[0] < img.shape[1]:
        img = img.transpose(1, 2, 0)
    
    if is_sar:
        band = img[:, :, 0] if img.ndim == 3 else img
    else:
        band = img[:, :, :3].astype(np.float32)
        
    band = np.nan_to_num(band)
    vmin, vmax = np.percentile(band, (2, 98))
    normalized = np.clip((band - vmin) / (vmax - vmin + 1e-5) * 255, 0, 255).astype(np.uint8)
    return normalized

# --- UI LAYOUT ---
uploaded_file = st.file_uploader("Drop a Sentinel-1 (.tif) file here...", type=["tif"])

if uploaded_file is not None:
    # 1. Process and display the uploaded query radar image
    sar_display = process_tif(uploaded_file, is_sar=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📥 Your Uploaded SAR Query")
        st.image(sar_display, caption="Sentinel-1 Radar Input", use_container_width=True, channels="GRAY")
    
    # 2. Extract features from the uploaded image
    with st.spinner("Analyzing radar textures and scanning vector space..."):
        # We need a 3-channel version for ResNet feature extraction
        sar_3ch = np.stack([sar_display]*3, axis=-1)
        query_vector = extractor.extract_vector(sar_3ch)
        
        # 3. Calculate Cosine Similarity against all 50 database entries
        # Formula: (A dot B) / (norm(A) * norm(B))
        dot_product = np.dot(db_vectors, query_vector)
        norms = np.linalg.norm(db_vectors, axis=1) * np.linalg.norm(query_vector)
        similarities = dot_product / (norms + 1e-8)
        
        # Get the index of the best matching vector
        best_match_idx = np.argmax(similarities)
        best_match_name = db_filenames[best_match_idx]
        score = similarities[best_match_idx]
        
    # 4. Display the best matching color image side-by-side
    best_match_path = os.path.join(S2_FOLDER, best_match_name)
    
    with col2:
        st.subheader(f"🎯 Best Database Match (Confidence: {score:.2%})")
        if os.path.exists(best_match_path):
            opt_display = process_tif(best_match_path, is_sar=False)
            st.image(opt_display, caption=f"Retrieved: {best_match_name}", use_container_width=True)
        else:
            st.warning(f"Found match {best_match_name}, but the full file isn't in your S2 folder path.")