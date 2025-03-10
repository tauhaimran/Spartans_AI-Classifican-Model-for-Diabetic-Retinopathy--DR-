## 💻 Installation & Setup

### 📁 Files Included
- `requirements.txt` - List of all required packages.
- `trained_model.h5` / `trained_model.keras` - The trained model files.

---

### 1. Upload Files to Google Colab
- Open [Google Colab](https://colab.research.google.com/)
- Upload the following files to the Colab session:
  - `model_training.ipynb`
  - `trained_model.h5` & `trained_model.keras`)

### 2. Open `model_training.ipynb`
- Make sure it appears under your Colab Files.

### 3. Install Dependencies (If Needed)
Run the following commands in a new code cell:
```bash
!pip install -r requirements.txt
!pip install tensorflow
```

### 4. Run the Last Cell Only
- In the `model_training.ipynb` file, **scroll down and run the last cell** only. 
- This will load the trained model and allow you to make predictions.

---

## 📌 Important Note
Ensure your `trained_model.h5` or `trained_model.keras` file is uploaded to the Colab session before running the last cell.
