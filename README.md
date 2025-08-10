#  Wearable-Based Cardiovascular Risk Predictor

A **machine learning and AI-powered application** that predicts **cardiovascular risk** based on **wearable device health data**.  
It uses deep learning (TensorFlow/Keras) and SHAP explainability, with an interactive UI built using **Streamlit**.

---

## 📜 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 📌 Overview
Wearable devices collect valuable health data such as heart rate, blood oxygen levels, ECG readings, skin temperature, and sleep patterns.  
This project leverages that data to **predict the risk of cardiovascular issues** using a trained deep learning model and provides **feature importance insights** using SHAP.

The app can:
- Accept manual health data input.
- Predict cardiovascular risk.
- Display results using **interactive charts**.
- Show which features contributed most to the prediction.

---

## ✨ Features
- 📊 **Manual Input Form** for wearable metrics.
- ⚡ **Real-Time Risk Prediction** with deep learning.
- 📈 **Feature Importance Visualization** (SHAP values).
- 🎯 **Interactive Gauge Chart** for easy risk interpretation.

---

## 📂 Dataset
We used the **[Wearables Dataset from Kaggle](https://www.kaggle.com/datasets/manideepreddy966/wearables-dataset)** containing synthetic wearable health data.

Example features used:
- Heart Rate
- Blood Oxygen Level
- ECG
- Skin Temperature
- Sleep Duration
- Stress Level

---

## 📁 Project Structure
```plaintext
wearable_cvd_app/
├── app.py                        # Streamlit application
├── train_model.py                 # Script for model training
├── requirements.txt               # Project dependencies
├── models/
│   ├── wearable_risk_model.h5     # Trained model file
│   ├── wearable_scaler.pkl        # Feature scaler
│   └── features.csv               # Feature names
├── data/
│   └── wearable_dataset.csv       # Dataset file
└── README.md
```


## ⚙ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/wearable-cvd-risk.git
cd wearable-cvd-risk
```

### 2️⃣ Create a virtual environment (Recommended)

```bash
python -m venv venv
```

Activate it:
**Windows**:

```bash
venv\Scripts\activate
```

**Mac/Linux**:

```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the app:

```bash
streamlit run app.py
```

Then open:

```bash
http://localhost:8501
```

---

## 🧠 Model Training

To train the model on the dataset:

```bash
python train_model.py
```

Generated files:

```bash
models/wearable_risk_model.h5
models/wearable_scaler.pkl
models/features.csv
```

---

## 🛠 Tech Stack

```bash
Python 3.x
TensorFlow / Keras
scikit-learn
pandas
numpy
SHAP
Streamlit
Plotly / Matplotlib
```

---

## 📜 License

```bash
This project is intended for educational purposes only and should not be used for actual medical diagnosis.
```

