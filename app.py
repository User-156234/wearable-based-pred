# app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Wearable Cardiovascular Risk Predictor", layout="centered")

# -------------------------
# Load model + scaler
# -------------------------
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("models/wearable_risk_model.h5")
    scaler = joblib.load("models/wearable_scaler.pkl")
    try:
        features = pd.read_csv("personal_health_data.csv")["feature"].tolist()
    except:
        features = ["Heart_Rate", "Blood_Oxygen_Level", "ECG", "Skin_Temperature", "Sleep_Duration", "Stress_Level"]
    return model, scaler, features

model, scaler, features = load_model_and_scaler()

# -------------------------
# Title & info
# -------------------------
st.title("Wearable-Based Cardiovascular Risk Predictor")
st.markdown("Enter your wearable health data to predict cardiovascular risk.")

st.write("⚠ Please note that this is a simulated model and actual results may vary.")

# -------------------------
# Manual Input Form
# -------------------------
with st.form("input_form"):
    cols = st.columns(3)
    hr = cols[0].number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=75)
    spo2 = cols[1].number_input("Blood Oxygen (%)", min_value=70.0, max_value=100.0, value=97.0)
    ecg_sel = cols[2].selectbox("ECG", options=["Normal", "Abnormal"], index=0)
    skin_temp = cols[0].number_input("Skin Temperature (°C)", value=32.5, step=0.1)
    sleep = cols[1].number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    stress_sel = cols[2].selectbox("Stress Level", options=["Low", "Moderate", "High"], index=1)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    # -------------------------
    # Prepare input
    # -------------------------
    ecg_map = {"Normal": 1, "Abnormal": 0}
    stress_map = {"Low": 0, "Moderate": 1, "High": 2}
    ecg = ecg_map[ecg_sel]
    stress = stress_map[stress_sel]

    input_list = [hr, spo2, ecg, skin_temp, sleep, stress]
    input_arr = np.array(input_list).reshape(1, -1)
    input_scaled = scaler.transform(input_arr)
    input_3d = input_scaled.reshape((1, 1, input_scaled.shape[1]))

    # -------------------------
    # Predict
    # -------------------------
    pred = model.predict(input_3d, verbose=0)[0][0]
    risk = pred * 100

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text': "Cardiovascular Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if risk >= 50 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Message
    if pred >= 0.5:
        st.error(f"⚠ High Cardiovascular Risk: {risk:.2f}%")
    else:
        st.success(f"✅ Low Cardiovascular Risk: {risk:.2f}%")

    # -------------------------
    # SHAP feature importance
    # -------------------------
    st.subheader("Feature Contribution to Prediction")

    # Try loading actual training data background for better SHAP results
    try:
        train_data = pd.read_csv("wearables_train_data.csv")  # Save during training
        background_data = train_data[features].values
        background_scaled = scaler.transform(background_data)
        background = background_scaled[np.random.choice(background_scaled.shape[0], 50, replace=False)]
    except:
        background = np.tile(scaler.mean_.reshape(1, -1), (50, 1))

    def model_predict(data_2d):
        data_3d = data_2d.reshape(data_2d.shape[0], 1, data_2d.shape[1])
        preds = model.predict(data_3d, verbose=0)
        return preds.reshape(-1,)

    explainer = shap.KernelExplainer(model_predict, background)
    shap_vals = explainer.shap_values(input_scaled, nsamples=300)
    sv = np.array(shap_vals).reshape(-1,)
    abs_vals = np.abs(sv)

    # Normalize values for better visualization
    if abs_vals.sum() > 0:
        abs_vals = abs_vals / abs_vals.sum()

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, abs_vals, align='center', color='skyblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("|SHAP value| (normalized)")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    st.caption("Note: SHAP values are approximate. For production, use larger background data and higher nsamples.")
