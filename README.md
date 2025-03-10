# Diabetic Retinopathy Classification Model Report

## 📌 Introduction
This project aims to build a **Diabetic Retinopathy (DR) Classification Model** that can identify the severity of DR from retinal images. The categories used for classification are:
- **No DR**
- **Mild**
- **Moderate**
- **Severe**
- **Proliferative**

The model is trained using a deep learning approach, leveraging popular frameworks like **TensorFlow, PyTorch, OpenCV, and FastAI**.

---

## 🔍 Approach
### 1. **Dataset Acquisition**
The dataset is obtained from Kaggle using the `kagglehub` library:
- Dataset URL: [Kaggle](https://www.kaggle.com/kushagratandon12/diabetic-retinopathy-balanced)
- The dataset is downloaded and extracted to a specified path.

### 2. **Data Preprocessing**
- The images are resized to a standard size (`224x224`) for compatibility with popular pre-trained models.
- The images are normalized to improve model performance.

### 3. **Model Selection & Training**
- Various models from **TensorFlow’s Keras API** are utilized:
  - **VGG16**
  - **VGG19**
  - **InceptionV3**
  - **ResNet50**
- Transfer learning is applied by loading pre-trained models and adding custom classification layers.
- **Callbacks** used during training include:
  - **EarlyStopping** - Stops training when a monitored metric has stopped improving.
  - **ReduceLROnPlateau** - Reduces learning rate when a metric has stopped improving.
  - **ModelCheckpoint** - Saves the best model during training.
- Optimizer used: **Adam**
- Loss function: **Categorical Cross-Entropy**

### 4. **Evaluation & Testing**
- The model is evaluated on a separate testing dataset.
- Predictions are displayed with corresponding class labels.

---

## 📈 Results
- The model achieves satisfactory accuracy in predicting the severity of Diabetic Retinopathy.
- The final trained model is saved as `trained_model.h5`.

---

## 🚀 Running the Model
To run the model, use the provided `run_model.py` file and follow the installation instructions in the `README.md` file.

---

## 📜 Conclusion
The approach used in this project effectively classifies Diabetic Retinopathy images into different severity levels. The model can be further improved by experimenting with:
- **More advanced architectures**.
- **Data augmentation techniques**.
- **Ensembling different models**.

