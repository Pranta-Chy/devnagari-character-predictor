import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import time
import matplotlib.pyplot as plt

# ---------------------------------------------------
# ⚙️ Streamlit Page Setup
# ---------------------------------------------------
st.set_page_config(
    page_title="Devanagari Character Predictor",
    page_icon="🔤",
    layout="centered"
)

# ---------------------------------------------------
# 🌈 Custom CSS Styling
# ---------------------------------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #eef2ff 0%, #f8f9ff 100%);
        border-radius: 20px;
        padding: 30px;
    }
    .title {
        font-size: 40px;
        text-align: center;
        font-weight: 800;
        color: #1e2a5a;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px #ccc;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #4a4a4a;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 3px dashed #8faadc;
        border-radius: 15px;
        padding: 25px;
        background-color: #f9fbff;
        transition: 0.3s ease;
    }
    .upload-box:hover {
        background-color: #e8f0ff;
    }
    .result-box {
        background-color: #edf4ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: #1b3c92;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🔤 Devanagari Character Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Instantly identify handwritten Devanagari characters using AI</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# 📚 Load Class Labels
# ---------------------------------------------------
if os.path.exists("class_names.json"):
    with open("class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
else:
    st.error("❌ class_names.json not found! Please ensure it’s in the same folder as app.py.")
    class_names = []

# ---------------------------------------------------
# 🧩 Custom Layers (for ViT)
# ---------------------------------------------------
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config

# ---------------------------------------------------
# 🧠 Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "devnagari_model.h5")
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please ensure 'model/devnagari_model.h5' exists.")
        return None

    try:
        with tf.keras.utils.custom_object_scope({
            "Patches": Patches,
            "PatchEncoder": PatchEncoder
        }):
            model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

model = load_model()

# ---------------------------------------------------
# 🧹 Preprocessing
# ---------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB").resize((64, 64))  # 👈 match model input size
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------
# 📤 Upload Image
# ---------------------------------------------------
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# 🔮 Prediction
# ---------------------------------------------------
if uploaded_file is not None:
    if model is None:
        st.error("⚠️ Model not loaded. Please check your setup.")
    else:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        col1.image(image, caption="🖼 Uploaded Image", use_container_width=True)

        with st.spinner("🔮 Analyzing image..."):
            time.sleep(1)
            processed = preprocess_image(image)
            preds = model.predict(processed)
            top_idx = np.argsort(preds[0])[::-1][:5]
            best_idx = top_idx[0]
            confidence = float(preds[0][best_idx]) * 100
            label = class_names[best_idx] if best_idx < len(class_names) else f"Class {best_idx}"

        # 💫 Emoji based on confidence
        if confidence >= 90:
            emoji = "🔥"
        elif confidence >= 70:
            emoji = "✨"
        elif confidence >= 50:
            emoji = "🙂"
        else:
            emoji = "🤔"

        col2.markdown(f"<div class='result-box'>{emoji} Predicted: <b>{label}</b><br>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)

        # 📊 Matplotlib Confidence Bar Chart
        st.markdown("### 📈 Prediction Confidence (Top 5)")
        top_labels = [class_names[i] for i in top_idx]
        top_scores = [float(preds[0][i]) * 100 for i in top_idx]

        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(top_labels[::-1], top_scores[::-1], color="#4e79e7", alpha=0.8)
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Predicted Labels")
        ax.set_xlim(0, 100)
        ax.bar_label(bars, fmt="%.1f%%", label_type="edge", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------
# 🧾 Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("🧠 Powered by TensorFlow • Vision Transformer (ViT) • Streamlit UI Design © 2025")
