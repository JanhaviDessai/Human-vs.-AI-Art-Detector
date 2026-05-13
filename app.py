import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras import layers, models, optimizers

# --- 1. SETTINGS ---
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

st.set_page_config(page_title="AI Detector (Integrated)", layout="centered")
st.title("🕵️‍♂️ AI vs Real: Integrated Backend")

# --- 2. THE INTEGRATED TRAINING BACKEND ---
@st.cache_resource
def train_and_load_model():
    """
    This function performs the 'ml_da.py' logic inside the app.
    It only runs ONCE due to @st.cache_resource.
    """
    st.info("🔄 Backend: Initializing MobileNetV2 and training... (This happens only once)")
    
    # Define Architecture (from your ml_da.py)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False 

    model = models.Sequential([
        layers.Input(shape=(160, 160, 3)),
        layers.Rescaling(1./127.5, offset=-1), 
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # --- LOADING DATA & TRAINING ---
    # Note: You MUST have the 'cifake_data' folder in your directory for this to work
    if os.path.exists('cifake_data'):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            'cifake_data/train',
            validation_split=0.8,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='binary'
        )
        # Fast training (1 epoch just to get the backend running)
        # In a real scenario, you'd want more epochs, but that takes time!
        model.fit(train_ds, epochs=1) 
        st.success("✅ Backend: Training complete!")
    else:
        st.error("❌ 'cifake_data' folder not found! Please download the dataset to train the backend.")
        st.stop()
        
    return model

# Initialize the backend
model = train_and_load_model()

# --- 3. FRONTEND ---
uploaded_file = st.file_uploader("Upload an image for the backend to classify", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Target Image', use_container_width=True)
    
    img_resized = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_batch)
    score = prediction[0][0]
    
    if score >= 0.5:
        st.success(f"Result: REAL (Score: {score:.2%})")
    else:
        st.error(f"Result: AI GENERATED (Score: {(1-score):.2%})")