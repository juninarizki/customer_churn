import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# --- 1. Load Artifacts ---
@st.cache_resource
def load_artifacts():
    with open('label_encoders.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return le, sc, model

try:
    label_encoders, scaler, model = load_artifacts()
    feature_names = scaler.feature_names_in_
except Exception as e:
    st.error(f"Error loading models/artifacts: {e}")
    st.stop()

# --- 2. Streamlit UI ---
st.title("📊 Customer Churn Classifier")
st.write("Masukkan data pelanggan di bawah ini untuk memprediksi probabilitas churn.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        usage_freq = st.number_input("Usage Frequency", min_value=0, value=10)
        sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

    with col2:
        support_calls = st.number_input("Support Calls", min_value=0, value=2)
        pay_delay = st.number_input("Payment Delay (Days)", min_value=0, value=0)
        contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

    submit = st.form_submit_button("Prediksi Sekarang")

if submit:
    # --- 3. Preprocessing User Input ---
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Tenure': [tenure],
        'Usage Frequency': [usage_freq],
        'Support Calls': [support_calls],
        'Payment Delay': [pay_delay],
        'Subscription Type': [sub_type],
        'Contract Length': [contract]
    })

    # Apply Label Encoding
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    # Apply Scaling
    input_ready = input_data[feature_names]
    input_scaled = scaler.transform(input_ready)

    # --- 4. Prediction ---
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # --- 5. Display Results ---
    st.divider()
    if prediction == 1:
        st.error(f"Hasil Prediksi: **CHURN** (Akan Berhenti)")
    else:
        st.success(f"Hasil Prediksi: **LOYAL** (Tetap Berlangganan)")

    st.write(f"Probabilitas Churn: **{probability:.2%}**")
    st.progress(probability)
