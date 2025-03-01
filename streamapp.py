import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from web3 import Web3
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import logging
import os
import google.generativeai as genai

# Setup logging for tracking anomalies
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Google Gemini API
API_KEY = "AIzaSyBYQCY5JwJPrN_u5aA4I57PgIk7DWB5fug"  # In production, use st.secrets
genai.configure(api_key=API_KEY)

# Blockchain Configuration
BLOCKCHAIN_API_KEY = "e65ee1b207274346b6a586c24e43bb18"  # In production, use st.secrets
BLOCKCHAIN_URL = f"https://mainnet.infura.io/v3/{BLOCKCHAIN_API_KEY}"
w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_URL))

class BlockchainLedger:
    def __init__(self):
        self.verified_addresses = set([
            '0x742d35Cc6634C0532925a3b844Bc454e4438f44e', 
            '0x1aD91ee08f21bE3dE0BA2ba6918E714dA6B45836'
        ])
        self.w3 = w3
        
    def is_connected(self):
        return True

    def verify_address(self, address):
        return Web3.is_address(address) and address in self.verified_addresses

    def is_verified(self, address):
        return self.verify_address(address)

ledger = BlockchainLedger()

# Initialize session state for persistent data
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'blocked_accounts' not in st.session_state:
    st.session_state.blocked_accounts = set()
if 'anomalies_detected' not in st.session_state:
    st.session_state.anomalies_detected = []
if 'llm_insights' not in st.session_state:
    st.session_state.llm_insights = ""
if 'graph_img' not in st.session_state:
    st.session_state.graph_img = None

# Function to automatically block/freeze accounts
def block_account(address):
    if address not in st.session_state.blocked_accounts:
        st.session_state.blocked_accounts.add(address)
        logger.warning(f"Account {address} has been automatically blocked/frozen.")

# Simulate Real-Time Transaction
def simulate_real_time_transaction():
    address = random.choice([
        Web3.to_checksum_address(f"0x{random.randbytes(20).hex()}"),
        random.choice([
            '0x742d35Cc6634C0532925a3b844Bc454e4438f44e', 
            '0x1aD91ee08f21bE3dE0BA2ba6918E714dA6B45836'
        ])
    ])
    transaction = {
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
        'Geolocation': random.choice(['US', 'EU', 'Asia']),
        'Address': address
    }
    st.session_state.transaction_history.append(transaction)
    return transaction

# Feature Engineering Functions
def feature_engineering_real_time_fake(transaction_data, previous_data=None):
    time_diff = (transaction_data['Timestamp'] - previous_data['Timestamp']).total_seconds() / 60.0 if previous_data else 0
    return pd.DataFrame([{
        'Transaction Frequency': transaction_data['Sent tnx'] + transaction_data['Received Tnx'],
        'Average Sent Amount': transaction_data['avg val sent'],
        'Average Received Amount': transaction_data['avg val received'],
        'Unique Sent Addresses': transaction_data['Unique Sent To Addresses'],
        'Unique Received Addresses': transaction_data['Unique Received From Addresses'],
        'Transaction Time Consistency': transaction_data['Time Diff between first and last (Mins)'],
        'Time Diff between Transactions (Minutes)': time_diff
    }])

def feature_engineering_real_time_sybil(transaction_data, previous_data=None):
    time_diff = (transaction_data['Timestamp'] - previous_data['Timestamp']).total_seconds() / 60.0 if previous_data else 0
    return pd.DataFrame([{
        'Transaction Frequency': transaction_data['Sent tnx'] + transaction_data['Received Tnx'],
        'Average Sent Amount': transaction_data['avg val sent'],
        'Average Received Amount': transaction_data['avg val received'],
        'Unique Sent Addresses': transaction_data['Unique Sent To Addresses'],
        'Unique Received Addresses': transaction_data['Unique Received From Addresses'],
        'Transaction Time Consistency': transaction_data['Time Diff between first and last (Mins)'],
        'Time Diff between Transactions (Minutes)': time_diff,
        'Account Age': transaction_data['Account Age'],
        'Total Sent Transactions': transaction_data['Sent tnx'],
        'Total Received Transactions': transaction_data['Received Tnx'],
        'Device Fingerprint': transaction_data['Device Fingerprint'],
        'IP Address': transaction_data['IP Address'],
        'Geolocation': transaction_data['Geolocation']
    }])

def feature_engineering_real_time_bot(transaction_data, previous_data=None):
    time_diff = (transaction_data['Timestamp'] - previous_data['Timestamp']).total_seconds() / 60.0 if previous_data else 0
    variance = np.std([transaction_data['avg val sent'], transaction_data['avg val received']])
    frequency = transaction_data['Sent tnx'] + transaction_data['Received Tnx']
    return pd.DataFrame([{
        'Transaction Time Diff': transaction_data['Time Diff between first and last (Mins)'],
        'Transaction Amount Variance': variance,
        'Unique Sent Addresses': transaction_data['Unique Sent To Addresses'],
        'Bot Activity Indicator': frequency * variance * transaction_data['Unique Sent To Addresses']
    }])

# Load models
try:
    pipeline_fake = joblib.load('fake_identities_model.pkl')
    pipeline_sybil = joblib.load('sybil_attacks_model.pkl')
    pipeline_bot = joblib.load('bot_activity_model.pkl')
except:
    # If models don't exist, create dummy models
    st.warning("Models not found. Using dummy models for demonstration.")
    # Create a simple dummy model for demo purposes
    dummy_model = RandomForestClassifier(n_estimators=10)
    dummy_model.fit(np.array([[1, 2, 3, 4]]), np.array([0]))
    pipeline_fake = pipeline_sybil = pipeline_bot = dummy_model

# Alert Detection Logic with Automatic Blocking
def detect_alerts(transaction_data, previous_data=None):
    current_alerts = []
    address = transaction_data['Address']
    
    # Skip further processing if the account is already blocked
    if address in st.session_state.blocked_accounts:
        current_alerts.append(f"Account {address} is already blocked.")
        st.session_state.alerts.extend(current_alerts)
        return

    df = pd.DataFrame(st.session_state.transaction_history)
    
    # Check for potential Sybil Attack clusters based on IP or device fingerprint
    if len(df) > 1:
        similar_ip = df[df['IP Address'] == transaction_data['IP Address']]
        similar_device = df[df['Device Fingerprint'] == transaction_data['Device Fingerprint']]
        if len(similar_ip) > 3 or len(similar_device) > 3:
            current_alerts.append(f"Sybil Attack detected - Account {address} has been automatically blocked.")
            block_account(address)
    
    # Calculate features for all models
    features_fake = feature_engineering_real_time_fake(transaction_data, previous_data)
    features_sybil = feature_engineering_real_time_sybil(transaction_data, previous_data)
    features_bot = feature_engineering_real_time_bot(transaction_data, previous_data)
    
    # Fake Identity Detection (investigation only)
    freq = features_fake['Transaction Frequency'].iloc[0]
    avg_sent = features_fake['Average Sent Amount'].iloc[0]
    time_diff = features_fake['Time Diff between Transactions (Minutes)'].iloc[0]
    if freq > 5 and avg_sent < 1.0 and (time_diff < 5 or time_diff == 0):
        current_alerts.append(f"Fake Identity detected for account {address}. Investigation recommended.")
    
    # Bot Detection - automatically block account
    bot_indicator = features_bot['Bot Activity Indicator'].iloc[0]
    if bot_indicator > 50:
        current_alerts.append(f"Bot Activity detected - Account {address} has been automatically blocked.")
        block_account(address)
    
    # Model-based Predictions for Sybil and Bot
    try:
        if pipeline_fake.predict(features_fake)[0] == 1:
            current_alerts.append(f"Fake Identity (Model): Investigation recommended for account {address}.")
        if pipeline_sybil.predict(features_sybil)[0] == 1:
            current_alerts.append(f"Sybil Attack (Model) detected - Account {address} has been automatically blocked.")
            block_account(address)
        if pipeline_bot.predict(features_bot)[0] == 1:
            current_alerts.append(f"Bot Activity (Model) detected - Account {address} has been automatically blocked.")
            block_account(address)
    except Exception as e:
        st.error(f"Error in model prediction: {str(e)}")
    
    # Unverified address handling (not auto-blocked)
    if not ledger.is_verified(address):
        current_alerts.append(f"Unverified Address: Blockchain check failed for account {address}. Follow-up action recommended.")
    
    if current_alerts:
        st.session_state.alerts.extend(current_alerts)

# Graph-based Anomaly Detection Functions
def create_wallet_transactions():
    """Create wallet transaction data from existing transaction history"""
    transactions = []
    
    for tx in st.session_state.transaction_history:
        # Create source wallet
        src = tx['Address']
        
        # Create destination wallet(s) based on sent transactions
        for _ in range(min(tx['Sent tnx'], 3)):  # Limit to max 3 destinations for clarity
            dst = f"wallet_{random.randint(1000, 9999)}"
            amount = tx['avg val sent']
            transactions.append((src, dst, amount))
    
    wallets = list(set([t[0] for t in transactions] + [t[1] for t in transactions]))
    return wallets, transactions

def build_wallet_graph(transactions):
    """Create a directed graph from transaction data"""
    G = nx.DiGraph()
    for src, dst, amount in transactions:
        if G.has_edge(src, dst):
            # Sum up transaction amounts between the same wallets
            G[src][dst]['weight'] += amount
        else:
            G.add_edge(src, dst, weight=amount)
    return G

def extract_features(G):
    """Extract wallet features from the transaction graph"""
    features = {}
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        
        # Sum of incoming transaction amounts
        in_amount = sum(G[u][node]['weight'] for u in G.predecessors(node)) if in_deg > 0 else 0
        
        # Sum of outgoing transaction amounts
        out_amount = sum(G[node][v]['weight'] for v in G.successors(node)) if out_deg > 0 else 0
        
        features[node] = [in_deg, out_deg, in_amount, out_amount]
    return features

def detect_graph_anomalies(features, contamination=0.1):
    """Detect anomalous wallets using Isolation Forest"""
    nodes = list(features.keys())
    data = np.array(list(features.values()))
    
    if len(data) == 0:
        logger.warning("No data to analyze for anomalies")
        return []
    
    # Normalize features for better detection
    if np.std(data, axis=0).min() > 0:  # Avoid division by zero
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    clf = IsolationForest(random_state=42, contamination=contamination)
    clf.fit(data)
    preds = clf.predict(data)
    # -1 indicates anomaly
    anomalies = [nodes[i] for i, pred in enumerate(preds) if pred == -1]
    
    # Add newly detected anomalies to global list and block them
    for addr in anomalies:
        if addr not in st.session_state.anomalies_detected:
            st.session_state.anomalies_detected.append(addr)
            st.session_state.alerts.append(f"Graph Anomaly detected for account {addr}. Automatically blocked.")
            block_account(addr)
    
    return anomalies

def visualize_graph(G, anomalies):
    """Visualize the transaction graph with highlighted anomalies"""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw normal nodes
    normal_nodes = [node for node in G.nodes() if node not in anomalies]
    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color='green', alpha=0.8, node_size=300)
    
    # Draw anomalous nodes
    if anomalies:
        nx.draw_networkx_nodes(G, pos, nodelist=anomalies, node_color='red', alpha=0.8, node_size=300)
    
    # Draw edges with width proportional to transaction amount
    edge_widths = [min(G[u][v]['weight'] * 0.5, 3) for u, v in G.edges()]  # Cap the width
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, arrows=True)
    
    # Draw labels (only for smaller graphs)
    if len(G.nodes()) < 30:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Wallet Network with Detected Anomalies")
    plt.axis('off')
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

def get_llm_insights(alerts_list, anomalies_list):
    """Get security insights from Google Gemini based on alerts and anomalies"""
    try:
        prompt = "You are a blockchain security analyst. Analyze the following alerts and detected anomalies, and provide insights and recommendations.\n\n"
        prompt += "Alerts:\n" + "\n".join(str(a) for a in alerts_list) + "\n\n"
        prompt += "Detected Anomalies (Blocked Accounts): " + ", ".join(str(a) for a in anomalies_list) + "\n\n"
        prompt += "Provide a brief summary and recommendations:"
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error getting LLM insights: {e}")
        return f"Error getting AI insights: {str(e)}"

# Create heatmap figure
def create_heatmap(df):
    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data yet")
        return fig
    
    heatmap_data = df.groupby('Geolocation').agg({
        'Sent tnx': 'sum',
        'Received Tnx': 'sum'
    }).reset_index()
    heatmap_data['Total Tnx'] = heatmap_data['Sent tnx'] + heatmap_data['Received Tnx']
    fig = px.density_heatmap(
        heatmap_data, 
        x='Geolocation', 
        y='Total Tnx', 
        title='Transaction Activity Heatmap',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=300)
    return fig

# Create network figure
def create_network_graph(df):
    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data yet")
        return fig
    
    G = nx.Graph()
    for i, row in df.iterrows():
        G.add_node(row['Address'])
        if i > 0 and random.random() < 0.3:
            G.add_edge(df.iloc[i-1]['Address'], row['Address'])
    
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray')))
    
    # Color nodes by whether they are blocked
    node_colors = ['red' if node in st.session_state.blocked_accounts else 'blue' for node in G.nodes()]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, 
        mode='markers', 
        marker=dict(size=10, color=node_colors),
        text=list(G.nodes()),
        hoverinfo='text'
    ))
    fig.update_layout(title='User Interaction Network', showlegend=False, height=300)
    return fig

# Create time series figure
def create_time_series(df):
    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data yet")
        return fig
    
    df['Total Tnx'] = df['Sent tnx'] + df['Received Tnx']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], 
        y=df['Total Tnx'], 
        mode='lines+markers', 
        name='Transaction Frequency'
    ))
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], 
        y=df['avg val sent'], 
        mode='lines+markers', 
        name='Avg Sent Amount',
        yaxis='y2'
    ))
    fig.update_layout(
        title='Transaction Metrics Over Time',
        yaxis=dict(title='Transaction Frequency'),
        yaxis2=dict(title='Amount (ETH)', overlaying='y', side='right'),
        height=300
    )
    return fig

# Initialize the app with some data
def initialize_data():
    if not st.session_state.transaction_history:
        for _ in range(10):
            tx = simulate_real_time_transaction()
            previous = st.session_state.transaction_history[-2] if len(st.session_state.transaction_history) >= 2 else None
            detect_alerts(tx, previous)

# Run AI Analysis
def run_ai_analysis():
    alerts_text = st.session_state.alerts
    anomalies_text = list(st.session_state.blocked_accounts)
    
    with st.spinner("Running AI analysis..."):
        st.session_state.llm_insights = get_llm_insights(alerts_text, anomalies_text)

# Generate new transaction
def generate_transaction():
    previous_data = st.session_state.transaction_history[-1] if st.session_state.transaction_history else None
    transaction_data = simulate_real_time_transaction()
    detect_alerts(transaction_data, previous_data)
    st.experimental_rerun()

# Run graph anomaly detection
def run_graph_analysis():
    if len(st.session_state.transaction_history) > 10:
        with st.spinner("Analyzing wallet network..."):
            wallets, transactions = create_wallet_transactions()
            G = build_wallet_graph(transactions)
            features = extract_features(G)
            anomalies = detect_graph_anomalies(features)
            
            # Generate wallet graph visualization
            st.session_state.graph_img = visualize_graph(G, anomalies)
            st.experimental_rerun()
    else:
        st.warning("Need more transaction data for graph analysis. Generate more transactions.")

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Blockchain Transaction Monitoring",
        page_icon="🔗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize data if needed
    initialize_data()
    
    # Title and description
    st.title("Blockchain Transaction Monitoring Dashboard")
    st.markdown("Monitor and analyze blockchain transactions in real-time for security threats.")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("Generate New Transaction"):
        generate_transaction()
    
    if st.sidebar.button("Run Graph Analysis"):
        run_graph_analysis()
    
    if st.sidebar.button("Run AI Analysis"):
        run_ai_analysis()
    
    # Create dataframe from transaction history
    df = pd.DataFrame(st.session_state.transaction_history)
    
    # Main dashboard layout
    col1, col2 = st.columns([1, 3])
    
    # Column 1: Alerts and Blocked Accounts
    with col1:
        st.subheader("Alerts")
        alerts_container = st.container()
        
        with alerts_container:
            st.markdown("---") 
            for alert in st.session_state.alerts[-10:]:
                st.error(alert)
        
        st.subheader("Blocked Accounts")
        blocked_container = st.container()
        with blocked_container:
            for acc in sorted(st.session_state.blocked_accounts):
                st.warning(acc)
    
    # Column 2: Visualizations
    with col2:
        tab1, tab2, tab3 = st.tabs(["Heatmap", "Network Graph", "Time Series"])
        
        with tab1:
            st.plotly_chart(create_heatmap(df), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_network_graph(df), use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_time_series(df), use_container_width=True)
    
    # Wallet Graph Visualization
    st.subheader("Wallet Graph Anomaly Detection")
    if st.session_state.graph_img is not None:
        st.image(st.session_state.graph_img, caption="Wallet Network with Detected Anomalies", use_column_width=True)
    else:
        st.info("Run graph analysis to visualize the wallet network.")
    
    # AI Security Analysis
    st.subheader("AI Security Analysis")
    ai_container = st.container()
    with ai_container:
        if st.session_state.llm_insights:
            st.markdown(st.session_state.llm_insights)
        else:
            st.info("Click 'Run AI Analysis' to get security insights from the LLM.")
    
    # Transaction data table
    st.subheader("Transaction Data")
    if not df.empty:
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("No transaction data available.")

if __name__ == "__main__":
    main()