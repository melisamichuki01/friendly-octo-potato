"""Churn predictor: a Streamlit app wrapping a saved model.

Run locally:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()

st.title("Churn predictor")
st.write("Enter a customer's details in the sidebar, then click Predict.")

st.sidebar.header("Customer details")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly charge", 0.0, 200.0, 70.0)
contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)

row = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "Contract": contract,
}])

if st.button("Predict"):
    pred = model.predict(row)[0]
    st.metric("Prediction", "Will churn" if pred == 1 else "Will stay")
