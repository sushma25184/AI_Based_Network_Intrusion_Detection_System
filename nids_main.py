import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI NIDS Dashboard",
    layout="wide"
)

st.title("AI-Powered Network Intrusion Detection System")
st.markdown(
    """
    This system uses **Machine Learning (Random Forest)**  
    trained on the **CIC-IDS-2017 (Friday DDOS)** dataset  
    to detect **malicious network traffic**.
    """
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data/cicids2017.csv")
    df.columns = df.columns.str.strip()
    return df

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# --------------------------------------------------
# PREPROCESS DATA
# --------------------------------------------------
target_col = "Label"

data[target_col] = data[target_col].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

X = data.select_dtypes(include=[np.number])
y = data[target_col]

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# --------------------------------------------------
# SIDEBAR – MODEL CONTROLS
# --------------------------------------------------
# ✅ ADDED (DOES NOT CHANGE YOUR LOGIC)
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")

st.sidebar.header("Model Configuration")

train_size = st.sidebar.slider(
    "Training Data Size (%)", 60, 90, 80
)

n_trees = st.sidebar.slider(
    "Number of Trees", 50, 200, 100
)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
if st.sidebar.button("Train Model Now"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=(100 - train_size) / 100,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.success(f"Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --------------------------------------------------
# LIVE TRAFFIC SIMULATOR
# --------------------------------------------------
st.subheader("Live Traffic Simulator")

col1, col2 = st.columns(2)

with col1:
    flow_duration = st.number_input(
        "Flow Duration", value=500
    )
    total_packets = st.number_input(
        "Total Packets", value=100
    )

with col2:
    packet_length = st.number_input(
        "Packet Length Mean", value=500
    )
    active_mean = st.number_input(
        "Active Mean", value=50
    )

if st.button("Analyze Packet"):
    if flow_duration > 400 and total_packets > 80:
        st.error("🚨 MALICIOUS TRAFFIC DETECTED (DDOS)")
    else:
        st.success("✅ BENIGN TRAFFIC DETECTED")
