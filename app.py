# app.py 🚗 Streamlit - Car Price Prediction App

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("car_price_model .pkl") 
    scaler = joblib.load("scaler .pkl")          
    return model, scaler

model, scaler = load_model()

# Mappings for categorical features
fuel_map = {"Petrol": 2, "Diesel": 1, "CNG": 0}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 1, "Automatic": 0}

# Streamlit UI
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction")
st.markdown("### Enter the car details below:")

# Input fields
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, max_value=100.0, value=5.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
car_age = st.slider("Car Age (in years)", 0, 30, 5)

# Predict button
if st.button("Predict Selling Price"):
    fuel = fuel_map[fuel_type]
    seller = seller_map[seller_type]
    trans = trans_map[transmission]
    price_per_km = present_price / (kms_driven + 1)

    input_data = np.array([[present_price, kms_driven, fuel, seller, trans, owner, car_age, price_per_km]])
    scaled_data = scaler.transform(input_data)

    predicted_price = model.predict(scaled_data)[0]
    st.success(f"Estimated Selling Price: ₹ {round(predicted_price, 2)} Lakhs")
