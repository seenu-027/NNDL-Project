# 🧠 Handwritten Digit Recognition — MNIST PNG (CNN + Pygame Application)

A complete Deep Learning pipeline for handwritten digit classification using a **Convolutional Neural Network (CNN)** trained on the **MNIST PNG dataset**, along with a **real-time Pygame-based digit recognition application**.

<div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-13a000?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-blueviolet)

</div>

---

## 📌 **Overview**
This project covers the complete workflow of:

- ⭐ Training a **CNN** on MNIST (PNG version)  
- ⭐ Evaluating performance (Confusion Matrix + Precision–Recall Curve)  
- ⭐ Deploying a **Pygame-based interactive digit recognition app**  
- ⭐ Automated digit detection using OpenCV contouring  
- ⭐ Real-time prediction with confidence levels  

Perfect for **NNDL Course Projects**, **GitHub Portfolio**, and **DL Deployment Demos**.

---

## 📂 Dataset (MNIST PNG)

📥 **Download**  
https://www.kaggle.com/datasets/alexanderyyy/mnist-png/data

Expected structure:

```
mnist_png/
   train/
      0/ 1/ ... 9/
   test/
      0/ 1/ ... 9/
```

Place inside:

```
D:/NNDL PROJECT 2/archive/mnist_png
```

Or update paths inside:

- `training_code.ipynb`
- `evaluation.py`

---

## 📁 **Project Structure**

```
NNDL PROJECT 2/
│
├── archive/
│   └── mnist_png/
│
├── saved_images/
│
├── application.py
├── training_code.ipynb
├── evaluation.py
│
├── bestmodel_png.h5
│
├── confusion_matrices.png
├── precision_recall_curve.png
│
├── screen1.png – screen5.png
│
└── README.md
```

---

## ⭐ **Features**

### 🧠 **1. Deep Learning Training Pipeline**
- CNN with **BatchNorm**, **Dropout**, **Adam optimizer**
- Data Augmentation:
  - 🔄 Rotation  
  - 🔍 Zoom  
  - ↔ Shift  
  - 🌀 Shear  
- LR scheduling using `ReduceLROnPlateau`
- Automatic best model saving using `ModelCheckpoint`

---

### 📊 **2. Model Evaluation**
Outputs generated:

✔ Training Confusion Matrix  
✔ Testing Confusion Matrix  
✔ Precision–Recall Curve for digits **0–9**  
✔ Classification Report:
- Accuracy  
- Precision  
- Recall  
- F1-score  

---

### 🎮 **3. Real-Time Pygame Application**
Supports:

🖌 Draw digits  
📤 Upload digit images  
🔍 Automatic contour-based digit extraction  
🤖 CNN prediction with **confidence %**  
📸 Save screenshots  

Hotkeys:

| Key | Action |
|-----|--------|
| ENTER | Continue / Next |
| 1 | Drawing Mode |
| 2 | Upload Mode |
| S | Save Screenshot |
| C | Clear Screen |
| BACKSPACE | Go Back |
| Q | Quit |

---

## ⚙️ Installation & Setup

### 🐍 1. Install Python  
Use **Python 3.9 – 3.11**

https://www.python.org/downloads/

---

### 📦 2. Install Dependencies

(Optional) Virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install required libraries:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
tensorflow>=2.10
numpy
matplotlib
opencv-python
pygame
scikit-learn
```

---

## 🚀 Running the Project

### ▶️ **1. Train the Model**

Jupyter Notebook:

```
training_code.ipynb
```

Or script:

```bash
python training_code.py
```

Model saved as:

```
bestmodel_png.h5
```

---

### 🧪 **2. Evaluate Model**

```bash
python evaluation.py
```

Generated Files:

- `confusion_matrices.png`
- `precision_recall_curve.png`

---

### 🎮 **3. Launch Recognition App**

```bash
python application.py
```

---

## 🧠 Model Architecture (CNN)

```
Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout
Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout
Flatten → Dense → BatchNorm → Dropout
Dense (Softmax 10 classes)
```

- Optimizer: **Adam**  
- Epochs: **10**  
- Input: **28×28 grayscale**

---

## 🧰 Troubleshooting

### ⚠ Pygame window not opening
```bash
pip install pygame --upgrade
```

### ⚠ TensorFlow DLL import error
Install Microsoft Visual C++ Redistributable  
Check Python 3.9–3.11 compatibility  
(Optional) install GPU CUDA/cuDNN

---

## 🚀 Future Enhancements
- Flask/FastAPI Deployment  
- Real-time Webcam Digit Recognition  
- Mobile Deployment using **TFLite**  
- Upgrade to **ResNet / EfficientNet CNNs**  

---

## ❤️ Author
Designed for **NNDL – Deep Learning Project Submission**  
Fully documented & deployment-ready.

---
