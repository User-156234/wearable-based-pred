⌚ Wearable-Based Cardiovascular Risk Predictor
A machine learning project that predicts cardiovascular risk using wearable health data such as heart rate, SpO₂, ECG, skin temperature, sleep duration, and stress level.
The app provides risk prediction and feature importance visualization using Streamlit.

📌 Features
📊 Manual data entry for wearable health metrics.

⚡ Real-time cardiovascular risk prediction using a trained deep learning model.

📈 Feature importance visualization with SHAP values.

🎯 Interactive gauge chart for risk percentage.

📂 Project Structure
bash
Copy
Edit
wearable_cvd_app/
├── app.py                  # Streamlit web app
├── train_model.py          # Script to train the model
├── requirements.txt        # Python dependencies
├── models/
│   ├── wearable_risk_model.h5   # Trained model
│   ├── wearable_scaler.pkl      # Feature scaler
│   └── features.csv             # Feature names
├── data/
│   └── wearable_dataset.csv     # Dataset
└── README.md
⚙ Installation & Setup
1️⃣ Clone or Download this Repository
bash
Copy
Edit
git clone <repo_url>
cd wearable_cvd_app
2️⃣ Create a Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
Activate it:

Windows:

Copy
Edit
venv\Scripts\activate
Mac/Linux:

bash
Copy
Edit
source venv/bin/activate
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
📊 Training the Model
Run the training script to train your model on the wearable dataset:

bash
Copy
Edit
python train_model.py
This will create:

models/wearable_risk_model.h5

models/wearable_scaler.pkl

models/features.csv

🚀 Running the App
Start the Streamlit app:

bash
Copy
Edit
streamlit run app.py
The app will be available at:

arduino
Copy
Edit
http://localhost:8501
🛠 Tech Stack
Python

TensorFlow / Keras

scikit-learn

pandas

SHAP (Explainable AI)

Streamlit

Plotly / Matplotlib

📌 Notes
Categorical Encoding in training:

ECG: Normal → 1, Abnormal → 0

Stress Level: Low → 0, Moderate → 1, High → 2

For deployment, ensure model files are in the models/ folder.

📜 License
This project is for educational purposes only and should not be used for actual medical diagnosis.