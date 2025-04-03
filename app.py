import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # Add this import


st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("\U0001F3E6 Loan Eligibility Prediction App")

model_path = 'models/loan_model.pkl'

# Load the model
if not os.path.exists(model_path):
    st.error("\U0001F6A8 Model not found. Please run `train.py` to train and save the model.")
    st.stop()

model = joblib.load(model_path)

st.markdown("### \U0001F4DE Please fill in applicant details below:")

# ---- Inputs with defaults ----
gender = st.selectbox("Gender", ['Male', 'Female'], index=0)
married = st.selectbox("Married", ['Yes', 'No'], index=0)
dependents = st.selectbox("Number of Dependents", ['0', '1', '2', '3+'], index=0)
education = st.selectbox("Education", ['Graduate', 'Not Graduate'], index=0)
self_employed = st.selectbox("Self Employed", ['No', 'Yes'], index=0)
applicant_income = st.slider("Applicant Income", min_value=1000, max_value=25000, value=5000, step=500)
coapplicant_income = st.slider("Coapplicant Income", min_value=0, max_value=15000, value=1000, step=500)
loan_amount = st.slider("Loan Amount (in thousands)", min_value=50, max_value=700, value=150, step=10)
loan_term = st.selectbox("Loan Amount Term (in months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
credit_history = st.selectbox("Credit History", [1.0, 0.0], index=0)
property_area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'], index=0)

# ---- Button to predict ----
if st.button("\U0001F4CA Predict Loan Eligibility"):
    input_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Married_Yes': 1 if married == 'Yes' else 0,
        'Dependents_1': 1 if dependents == '1' else 0,
        'Dependents_2': 1 if dependents == '2' else 0,
        'Dependents_3+': 1 if dependents == '3+' else 0,
        'Education_Not Graduate': 1 if education == 'Not Graduate' else 0,
        'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
        'Property_Area_Rural': 1 if property_area == 'Rural' else 0,
        'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0
    }

    # Convert to DataFrame and align columns
    input_df = pd.DataFrame([input_data])

    # Align to model input
    model_features = model.feature_names_in_
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]
    prediction_label = "✅ Loan Approved" if prediction == 1 else "❌ Loan Denied"

    # Styled Result Box
    if prediction == 1:
        st.success(f"\U0001F4B8 {prediction_label}")
    else:
        st.error(f"\U0001F6AB {prediction_label}")

    # Visual summary
    st.subheader("\U0001F4C8 Feature Overview:")
    st.bar_chart(input_df.T.rename(columns={0: 'Value'}))

    # Pie Chart for Income Distribution
    labels = ['Applicant Income', 'Coapplicant Income']
    values = [applicant_income, coapplicant_income]
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig_pie.update_layout(title_text="Income Distribution")
    st.plotly_chart(fig_pie)

    # Histogram for Loan Amount
    fig_hist, ax = plt.subplots()
    ax.hist([loan_amount], bins=10, color='blue', alpha=0.7, label='Loan Amount')
    ax.set_xlabel("Loan Amount (in thousands)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig_hist)

    # Gauge Chart for Credit History
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=credit_history,
        title={'text': "Credit History"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "green" if credit_history == 1.0 else "red"},
            'steps': [
                {'range': [0, 1], 'color': "lightgray"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge)
