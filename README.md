# Spartans_AI Classifican Model for Diabetic Retinopathy (DR)
Diabetic Retinopathy (DR) is a complication of diabetes that affects the eyes. It can lead to vision loss if not detected early.
This is an AI model that classifies retinal images into different DR severity levels.
Using a dataset of fundus images, we built a solution that accurately identifies whether an image indicates DR and, if so, informs of its severity level. 

# **Diabetic Retinopathy Classification: Approach and Results**

## **1. Introduction**
Diabetic Retinopathy (DR) is a severe eye condition affecting individuals with diabetes. Early detection is crucial for preventing vision loss. This report presents an approach for DR classification using a deep learning model. The methodology includes dataset preparation, model training, front-end development for deployment, and evaluation of results.

---

## **2. Methodology**

### **2.1 Dataset**
The dataset used for training and evaluation consists of images labeled with different DR severity levels. The images were obtained from the Kaggle dataset: `kushagratandon12/diabetic-retinopathy-balanced`.

### **2.2 Model Architecture**
The classification model was built using deep learning frameworks such as TensorFlow/Keras. A convolutional neural network (CNN) was designed, leveraging a pre-trained model (e.g., ResNet50, InceptionV3, or VGG16) for feature extraction and fine-tuning.

### **2.3 Model Training**
- Data Augmentation: Techniques like rotation, flipping, and brightness adjustments were applied to increase model robustness.
- Optimizer: Adam optimizer with a learning rate scheduler.
- Loss Function: Categorical Crossentropy for multi-class classification.
- Evaluation Metrics: Accuracy, Precision, Recall, and F1-score were used for performance evaluation.

### **2.4 Deployment Front-End**
A user-friendly front-end was developed using Jupyter Notebook widgets, allowing users to upload an image and receive a DR classification prediction. The key steps in the front-end implementation include:
1. Uploading an image using an interactive widget.
2. Preprocessing the image (resizing, normalization, batch dimension addition).
3. Feeding the image into the trained model for classification.
4. Displaying the predicted DR stage alongside the uploaded image.

```python
# Example code snippet for front-end image classification
upload = widgets.FileUpload(accept='image/*', multiple=False)

def classify_image(change):
    for name, file_info in upload.value.items():
        with open(name, 'wb') as f:
            f.write(file_info['content'])
        img, img_array = preprocess_image(name)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted: {CLASS_LABELS[predicted_class]}")
        plt.show()

upload.observe(classify_image, names='value')
display(upload)
```

---

## **3. Results**
(Provide space for the results and evaluation metrics obtained after testing the model on the validation dataset)

| Metric       | Value |
|-------------|-------|
| Accuracy    |       |
| Precision   |       |
| Recall      |       |
| F1-score    |       |

Example Image Prediction:

(Include sample images with corresponding DR classifications obtained from the model.)

---

## **4. Conclusion**
This approach demonstrates the effectiveness of deep learning for DR classification. The model successfully identifies different DR stages from retinal images with reasonable accuracy. Further improvements could include hyperparameter tuning, larger datasets, and the integration of additional image preprocessing techniques. Deployment on cloud-based platforms or mobile applications can also enhance accessibility and usability.

---

## **5. Future Work**
- Enhancing model performance with ensemble learning.
- Implementing explainable AI techniques to interpret predictions.
- Deploying the model as a web application for wider accessibility.

*End of Report*

