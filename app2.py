import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load Saved Model & Scaler
# -----------------------------
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Customer Segmentation", page_icon="💳")

st.title("💳 Credit Card Customer Segmentation")
st.write("Enter customer details below to predict the customer segment.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📥 Customer Details")

balance = st.number_input("Balance", min_value=0.0, value=1000.0)
purchases = st.number_input("Purchases", min_value=0.0, value=500.0)
cash_advance = st.number_input("Cash Advance", min_value=0.0, value=200.0)
credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0)
payments = st.number_input("Payments", min_value=0.0, value=800.0)
purchase_frequency = st.slider("Purchase Frequency (0 to 1)", 0.0, 1.0, 0.5)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Segment"):

    # Prepare input data
    input_data = np.array([[balance,
                            purchases,
                            cash_advance,
                            credit_limit,
                            payments,
                            purchase_frequency]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    st.subheader("📊 Prediction Result")

    if cluster == 0:
        st.success("🔵 High Value Customer")

        st.markdown("""
        ### 📌 Customer Profile
        - High balance and high spending  
        - Large credit limit  
        - Active payment behavior  
        - Frequent credit usage  

        ### 💡 Business Recommendations
        - Offer premium credit cards  
        - Increase credit limits  
        - Provide loyalty rewards  
        - Personalized marketing offers  
        """)

    elif cluster == 1:
        st.info("🟠 Low / Moderate Customer")

        st.markdown("""
        ### 📌 Customer Profile
        - Lower spending activity  
        - Moderate credit usage  
        - Stable financial behavior  

        ### 💡 Business Recommendations
        - Encourage spending with cashback  
        - Promote EMI options  
        - Run engagement campaigns  
        """)

    st.divider()
    st.write(f"Cluster ID: {cluster}")

st.divider()

st.caption("Built using KMeans Clustering & PCA Visualization")