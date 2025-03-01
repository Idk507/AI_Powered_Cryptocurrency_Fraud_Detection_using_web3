import streamlit as st
import pandas as pd
import joblib

# Load the saved model and feature columns
pipeline = joblib.load('fraud_detection_model.pkl')
all_columns = joblib.load('feature_columns.pkl')

# Define the basic features to collect from the user
basic_features = [
    'Sent tnx',
    'Received Tnx',
    'avg val sent',
    'avg val received',
    'Unique Sent To Addresses',
    'Unique Received From Addresses',
    'Time Diff between first and last (Mins)'
]

# Define the questions corresponding to each feature
questions = [
    "How many transactions has this address sent? (Sent tnx)",
    "How many transactions has this address received? (Received Tnx)",
    "What is the average value sent in transactions? (avg val sent)",
    "What is the average value received in transactions? (avg val received)",
    "How many unique addresses has this address sent to? (Unique Sent To Addresses)",
    "How many unique addresses has this address received from? (Unique Received From Addresses)",
    "What is the time difference between the first and last transaction in minutes? (Time Diff between first and last (Mins))"
]

# Initialize session state to manage conversation flow
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'features' not in st.session_state:
    st.session_state.features = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to add messages to chat history
def add_to_chat(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

# Main application
st.title("Fraud Detection Chatbot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat logic
if st.session_state.step < len(questions):
    # Ask the next question
    question = questions[st.session_state.step]
    with st.chat_message("assistant"):
        st.markdown(question)
    add_to_chat("assistant", question)

    # Get user input
    user_input = st.chat_input("Your response:")

    if user_input:
        try:
            # Convert input to float and store it
            value = float(user_input)
            feature_name = basic_features[st.session_state.step]
            st.session_state.features[feature_name] = value
            add_to_chat("user", user_input)
            st.session_state.step += 1
        except ValueError:
            # Handle invalid input
            with st.chat_message("assistant"):
                st.markdown("Please enter a valid number.")
            add_to_chat("assistant", "Please enter a valid number.")
else:
    # All features collected, process the transaction
    with st.chat_message("assistant"):
        st.markdown("Thank you. Analyzing the transaction...")
    add_to_chat("assistant", "Thank you. Analyzing the transaction...")

    # Create transaction data with all columns initialized to 0
    transaction_data = {col: 0 for col in all_columns}
    transaction_data.update(st.session_state.features)

    # Perform feature engineering
    transaction_data['Transaction Frequency'] = transaction_data['Sent tnx'] + transaction_data['Received Tnx']
    transaction_data['Average Sent Amount'] = transaction_data['avg val sent']
    transaction_data['Average Received Amount'] = transaction_data['avg val received']
    transaction_data['Unique Sent Addresses'] = transaction_data['Unique Sent To Addresses']
    transaction_data['Unique Received Addresses'] = transaction_data['Unique Received From Addresses']
    transaction_data['Transaction Time Consistency'] = transaction_data['Time Diff between first and last (Mins)']
    # Add other engineered features if your model requires them

    # Create a dataframe for prediction
    transaction_df = pd.DataFrame([transaction_data])
    transaction_df = transaction_df[all_columns]  # Ensure column order matches training data

    # Make predictions
    fraud_prob = pipeline.predict_proba(transaction_df)[0, 1]
    prediction = pipeline.predict(transaction_df)[0]

    # Display the result
    result = f"The transaction has a fraud probability of {fraud_prob:.4f}."
    if prediction == 1:
        result += " ⚠️ Fraudulent Transaction Detected!"
    else:
        result += " Transaction appears legitimate."
    with st.chat_message("assistant"):
        st.markdown(result)
    add_to_chat("assistant", result)

    # Reset for the next transaction
    st.session_state.step = 0
    st.session_state.features = {}
    st.session_state.chat_history = []  # Clear chat history (optional)