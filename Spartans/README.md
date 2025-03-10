
## 💻 Installation & Setup

### 📁 Files Included
- `run_model.py` - The main Python file to run the model.
- `requirements.txt` - List of all required packages.
- `trained_model.h5` - The trained model file.

---


### 1. Clone the Repository
```bash
 git clone https://github.com/YourUsername/YourRepo.git
```

### 2. Navigate to the Project Directory
```bash
cd YourRepo
```

### 3. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv env
```
- **Windows:**
  ```bash
  .\env\Scripts\activate
  ```
- **Linux / MacOS:**
  ```bash
  source env/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Upgrade Pip (To avoid compatibility issues)
```bash
python -m pip install --upgrade pip
```

### 6. Install TensorFlow (If Not Installed Automatically)
```bash
pip install tensorflow
```
- For GPU support (if desired):
  ```bash
  pip install tensorflow-gpu
  ```

---

## 🚀 Running the Model
To run the model, use the following command:
```bash
python run_model.py
```

Upload an image when prompted and the model will display the prediction result along with the image.

---

## 📌 Important Note
Ensure your `dr_model.h5` file is placed in the **same directory** as `run_model.py`.



