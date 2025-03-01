import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# 1. Historical Data Preparation and Feature Engineering
# ------------------------------

@st.cache(allow_output_mutation=True)
def load_and_train_model():
    # Load the historical dataset
    file_path = "transaction_dataset.csv"
    df = pd.read_csv(file_path)
    
    # Drop irrelevant columns (adjust if needed)
    df = df.drop(columns=['Unnamed: 0', 'Index', 'Address'])
    
    # Define target and raw features
    y = df['FLAG']  # FLAG: 1 = Bot, 0 = Legitimate
    X = df.drop(columns=['FLAG'])
    
    # --- Feature Engineering for Bot Detection ---
    def feature_engineering_for_bot_detection(df):
        # Feature 1: Transaction Timing Consistency
        if 'Time Diff between first and last (Mins)' in df.columns:
            df['Transaction Time Diff'] = df['Time Diff between first and last (Mins)']
        else:
            df['Transaction Time Diff'] = 0

        # Feature 2: Unusual Transaction Amounts
        if set(['avg val sent', 'avg val received']).issubset(df.columns):
            df['Transaction Amount Variance'] = df[['avg val sent', 'avg val received']].std(axis=1)
        else:
            df['Transaction Amount Variance'] = 0

        # Feature 3: Pattern Consistency (unique addresses)
        if 'Unique Sent To Addresses' in df.columns:
            df['Unique Sent Addresses'] = df['Unique Sent To Addresses']
        else:
            df['Unique Sent Addresses'] = 0

        # Bot Activity Indicator (for demo)
        df['Bot Activity Indicator'] = (
            df['Transaction Time Diff'] * df['Transaction Amount Variance'] * df['Unique Sent Addresses']
        )
        return df

    df = feature_engineering_for_bot_detection(df)
    
    # Define the training features.
    # (Your historical dataset may include many more columns. Here we use all columns after feature engineering.)
    X = df.drop(columns=['FLAG'])
    # Save training columns (order matters)
    training_columns = X.columns.tolist()

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Build preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Build the ML pipeline
    pipeline_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return pipeline_model, training_columns, numerical_cols, categorical_cols, acc, report

# Load and train (cached so this runs only once)
pipeline_model, training_columns, numerical_cols, categorical_cols, test_acc, class_report = load_and_train_model()

st.title("Real-Time Bot Detection Application")
st.write("This application flags potential bots by analyzing transaction behavior in real time.")

st.subheader("Model Performance on Historical Data")
st.write(f"**Test Accuracy:** {test_acc:.4f}")
st.text(class_report)

# ------------------------------
# 2. Real-Time Simulation Functions
# ------------------------------

def align_features_for_real_time(df_sim):
    """
    Ensure the simulation DataFrame has exactly the same columns as training data.
    For missing numerical columns, use 0; for missing categorical, use 'Unknown'.
    """
    for col in training_columns:
        if col not in df_sim.columns:
            if col in numerical_cols:
                df_sim[col] = 0
            else:
                df_sim[col] = 'Unknown'
    df_sim = df_sim[training_columns]
    return df_sim

def simulate_real_time_transaction():
    """
    Simulate a single real-time transaction with raw data.
    """
    transaction_data = {
        'Sent tnx': random.randint(1, 10),
        'Received Tnx': random.randint(1, 10),
        'avg val sent': random.uniform(0.01, 5.0),
        'avg val received': random.uniform(0.01, 5.0),
        'Unique Sent To Addresses': random.randint(1, 20),
        'Unique Received From Addresses': random.randint(1, 20),
        'Time Diff between first and last (Mins)': random.uniform(1, 120),
        'Timestamp': datetime.now()
    }
    return transaction_data

def feature_engineering_for_real_time_bot_detection(transaction_data, previous_data=None):
    """
    Compute derived features from a simulated transaction.
    """
    # Derived features based on raw transaction data
    transaction_frequency = transaction_data['Sent tnx'] + transaction_data['Received Tnx']
    average_sent_amount = transaction_data['avg val sent']
    average_received_amount = transaction_data['avg val received']
    unique_sent_addresses = transaction_data['Unique Sent To Addresses']
    
    # Use previous transaction timestamp to compute time difference (if available)
    if previous_data is not None:
        time_diff = (transaction_data['Timestamp'] - previous_data['Timestamp']).total_seconds() / 60.0
    else:
        time_diff = 0

    # Compute amount variance using sent and received values
    transaction_amount_variance = np.std([average_sent_amount, average_received_amount])
    
    # Build features dictionary
    features = {
        'Transaction Frequency': transaction_frequency,
        'Average Sent Amount': average_sent_amount,
        'Average Received Amount': average_received_amount,
        'Unique Sent Addresses': unique_sent_addresses,
        # For simulation, we use the time difference as a proxy for "Transaction Time Consistency"
        'Transaction Time Consistency': time_diff,
        # Include historical engineered features (if available)
        'Transaction Time Diff': time_diff,  
        'Transaction Amount Variance': transaction_amount_variance,
        'Bot Activity Indicator': transaction_frequency * transaction_amount_variance * unique_sent_addresses,
    }
    
    features_df = pd.DataFrame([features])
    features_df = align_features_for_real_time(features_df)
    return features_df

def real_time_bot_detection(num_transactions=10, delay=2):
    """
    Simulate real-time monitoring of transactions for bot detection.
    Updates the Streamlit app with each simulated transaction.
    """
    previous_data = None  # For computing time differences
    output_area = st.empty()  # Placeholder to update simulation messages
    
    simulation_messages = []
    for i in range(num_transactions):
        transaction_data = simulate_real_time_transaction()
        features_df = feature_engineering_for_real_time_bot_detection(transaction_data, previous_data)
        
        # Get prediction probabilities and prediction
        bot_prob = pipeline_model.predict_proba(features_df)[0, 1]
        prediction = pipeline_model.predict(features_df)[0]
        
        msg = f"**Transaction {i+1}:** Predicted Bot Activity Probability: {bot_prob:.4f}. "
        if prediction == 1:
            msg += "⚠️ **Bot Detected!**"
        else:
            msg += "Transaction appears **legitimate**."
        simulation_messages.append(msg)
        
        # Update the output area with all messages so far
        output_area.markdown("\n\n".join(simulation_messages))
        previous_data = transaction_data
        time.sleep(delay)
    
    output_area.markdown("\n\n".join(simulation_messages))
    st.success("Simulation Completed.")

# ------------------------------
# 3. Streamlit Interface Controls
# ------------------------------

st.subheader("Real-Time Bot Detection Simulation")
if st.button("Start Simulation"):
    real_time_bot_detection(num_transactions=10, delay=2)
