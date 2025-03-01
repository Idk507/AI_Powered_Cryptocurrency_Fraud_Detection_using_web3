import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
from datetime import datetime

# Load the saved model and feature columns
model = joblib.load("model.pkl")
training_columns = joblib.load("feature_columns.pkl")

# For Streamlit, define which expected columns are numerical (to set defaults)
numerical_cols = [col for col in training_columns if col not in ['Device Fingerprint', 'IP Address', 'Geolocation']]

def align_features(df_sim):
    """
    Ensure the simulation DataFrame has exactly the same columns (and order)
    as the training data. Missing numerical columns are set to 0;
    missing categorical columns to 'Unknown'.
    """
    for col in training_columns:
        if col not in df_sim.columns:
            df_sim[col] = 0 if col in numerical_cols else 'Unknown'
    return df_sim[training_columns]

def simulate_real_time_transaction():
    """
    Simulate a single real-time transaction with raw input data.
    """
    transaction_data = {
        'Sent tnx': random.randint(1, 10),
        'Received Tnx': random.randint(1, 10),
        'avg val sent': random.uniform(0.01, 5.0),
        'avg val received': random.uniform(0.01, 5.0),
        'Unique Sent To Addresses': random.randint(1, 20),
        'Unique Received From Addresses': random.randint(1, 20),
        'Time Diff between first and last (Mins)': random.uniform(1, 120),
        'Timestamp': datetime.now(),
        'Account Age': random.randint(1, 100),
        'Device Fingerprint': random.choice(['DeviceA', 'DeviceB', 'DeviceC']),
        'IP Address': random.choice(['192.168.0.1', '192.168.0.2']),
        'Geolocation': random.choice(['US', 'EU', 'Asia'])
    }
    return transaction_data

def feature_engineering_for_real_time(transaction_data, previous_data=None):
    """
    Build a feature vector from raw transaction data.
    Computes derived features and aligns the result with training_columns.
    """
    # Derived features
    transaction_frequency = transaction_data['Sent tnx'] + transaction_data['Received Tnx']
    average_sent_amount = transaction_data['avg val sent']
    average_received_amount = transaction_data['avg val received']
    unique_sent_addresses = transaction_data['Unique Sent To Addresses']
    unique_received_addresses = transaction_data['Unique Received From Addresses']
    time_diff_first_last = transaction_data['Time Diff between first and last (Mins)']
    
    # Compute time difference between transactions if previous data is available
    if previous_data is not None:
        previous_time = previous_data['Timestamp']
        time_diff = (transaction_data['Timestamp'] - previous_time).total_seconds() / 60.0
    else:
        time_diff = 0

    features = {
        'Transaction Frequency': transaction_frequency,
        'Average Sent Amount': average_sent_amount,
        'Average Received Amount': average_received_amount,
        'Unique Sent Addresses': unique_sent_addresses,
        'Unique Received Addresses': unique_received_addresses,
        'Transaction Time Consistency': time_diff_first_last,
        'Time Diff between Transactions (Minutes)': time_diff,
        'Account Age': transaction_data['Account Age'],
        'Total Sent Transactions': transaction_data['Sent tnx'],
        'Total Received Transactions': transaction_data['Received Tnx'],
        'Device Fingerprint': transaction_data['Device Fingerprint'],
        'IP Address': transaction_data['IP Address'],
        'Geolocation': transaction_data['Geolocation']
    }
    features_df = pd.DataFrame([features])
    features_df = align_features(features_df)
    return features_df

# Streamlit App Interface
st.title("Real-Time Sybil Attack Detection Simulation")

if st.button("Simulate Transaction"):
    transaction_data = simulate_real_time_transaction()
    features_df = feature_engineering_for_real_time(transaction_data)
    
    st.write("### Simulated Transaction Features")
    st.write(features_df)
    
    # Predict using the loaded model
    prob = model.predict_proba(features_df)[0, 1]
    pred = model.predict(features_df)[0]
    
    st.write(f"**Predicted Sybil Attack Probability:** {prob:.4f}")
    if pred == 1:
        st.error("⚠️ Sybil Attack Detected! Transaction flagged.")
    else:
        st.success("Transaction appears legitimate.")
