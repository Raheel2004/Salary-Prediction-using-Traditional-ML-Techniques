# Salary-Prediction-using-Traditional-ML-Techniques
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and encoders
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Salary Prediction App")

# User Inputs
age = st.slider("Age", 20, 60)
experience = st.slider("Experience", 0, 40)
education = st.selectbox("Education Level", label_encoders['Education'].classes_)
skill = st.selectbox("Skill Level", label_encoders['Skill Level'].classes_)

# Encode input
input_df = pd.DataFrame({
    'Age': [age],
    'Experience': [experience],
    'Education': [label_encoders['Education'].transform([education])[0]],
    'Skill Level': [label_encoders['Skill Level'].transform([skill])[0]]
})

input_df[['Age', 'Experience']] = scaler.transform(input_df[['Age', 'Experience']])

# Predict
prediction = model.predict(input_df)[0]
st.success(f"Predicted Salary: ${prediction:,.2f}")
