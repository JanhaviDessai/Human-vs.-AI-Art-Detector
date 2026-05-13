# Human vs AI Art Detector

A deep learning-powered image classification system that detects whether an image is AI-generated or human-created using MobileNetV2 and TensorFlow. The project includes a Streamlit-based frontend for real-time image uploads and predictions.

---

## Overview

With the rise of Generative AI tools, distinguishing AI-generated art from authentic human-created artwork has become increasingly challenging. This project uses Transfer Learning with MobileNetV2 to classify uploaded images into:

- Human-Created / Real
- AI-Generated

The application combines machine learning with an interactive web interface to provide fast and accurate image classification.

---

## Features

- AI vs Human image classification
- Transfer Learning using MobileNetV2
- Streamlit-powered web interface
- Real-time image prediction
- TensorFlow/Keras deep learning pipeline
- Optimized image preprocessing workflow
- Lightweight and deployment-ready architecture

---

## Tech Stack

### Languages & Libraries
- Python
- TensorFlow / Keras
- NumPy
- PIL (Python Imaging Library)

### Frameworks & Tools
- Streamlit
- MobileNetV2
- Jupyter Notebook
- Google Colab

---

## Model Architecture

The project uses MobileNetV2 as the backbone model for transfer learning.

### Architecture Flow
1. Image Input (160x160)
2. Image Normalization & Rescaling
3. MobileNetV2 Feature Extraction
4. Global Average Pooling
5. Dropout Layer
6. Dense Sigmoid Output Layer

### Training Configuration
- Pretrained ImageNet weights
- Frozen base layers
- Adam optimizer
- Binary Crossentropy loss
- Binary classification output

---

## Dataset

The model is trained on the CIFAKE dataset containing:
- AI-generated images
- Real human-created images

Expected dataset structure:

```bash
cifake_data/
│
├── train/
│   ├── REAL/
│   └── FAKE/
```

---

## Files Included

### ML_DA.ipynb
Contains:
- Data preprocessing
- Dataset loading
- Model building
- Model training
- Performance evaluation

### app.py
Contains:
- Streamlit frontend
- Image upload interface
- Model loading and inference
- Real-time prediction system

---

## Installation & Setup

### 1. Install Dependencies

```bash
pip install tensorflow streamlit pillow numpy
```

### 2. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## How It Works

1. User uploads an image
2. Image is resized to 160x160
3. MobileNetV2 extracts image features
4. Model predicts classification probability
5. System displays:
   - REAL
   - AI GENERATED

---

## Learning Outcomes

- Transfer Learning with MobileNetV2
- Deep Learning for image classification
- TensorFlow model deployment
- Streamlit frontend development
- Image preprocessing and inference
- Real-time ML application integration

---

## Future Improvements

- Improve model accuracy with larger datasets
- Add confidence score visualizations
- Implement Grad-CAM explainability
- Add batch image prediction
- Deploy on cloud platforms
- Fine-tune MobileNetV2 layers

---

## Author

Janhavi Naik Dessai  
Data Science Undergraduate  
Generative AI & Machine Learning Enthusiast

GitHub: github.com/JanhaviDessai
