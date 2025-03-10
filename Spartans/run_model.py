import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ipywidgets as widgets
from tensorflow.keras.preprocessing import image
from IPython.display import display
from PIL import Image
import torch

# Load your trained model
MODEL_PATH = "model/trained_model.h5"  # Change if using PyTorch
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASS_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

print("Upload file into model... ")
# Image upload widget
upload = widgets.FileUpload(accept='image/*', multiple=False)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))  # Adjust based on model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img, img_array

def classify_image(change):
    for name, file_info in upload.value.items():
        with open(name, 'wb') as f:
            f.write(file_info['content'])
        
        # Preprocess image
        img, img_array = preprocess_image(name)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Show image and result
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted: {CLASS_LABELS[predicted_class]}")
        plt.show()

upload.observe(classify_image, names='value')
display(upload)
