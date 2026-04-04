import streamlit as st
import pandas as pd
import pickle
import numpy as np


# 1. Load the saved model and columns
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.title("📊 Professional Churn Predictor")
st.write("This app uses a Random Forest model to predict customer behavior.")

# 2. User Inputs
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 50)
total_charges = tenure * monthly_charges # Simple estimation

# 3. Prepare the data for the model
# We create a dictionary with all 0s, then fill in the values we have
input_data = pd.DataFrame(0, index=[0], columns=model_columns)
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges

# 4. Make Prediction
prediction_proba = model.predict_proba(input_data)[0][1]

# 5. Show Results
st.subheader("Analysis")
if prediction_proba > 0.5:
    st.error(f"⚠️ High Risk: {prediction_proba:.1%} chance of leaving.")
    st.write("Strategy: Offer a loyalty discount or contract extension.")
else:
    st.success(f"✅ Low Risk: {prediction_proba:.1%} chance of leaving.")
    st.write("Strategy: Customer is satisfied. Consider upselling premium services.")