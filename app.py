import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="Anomaly Detection App", layout="wide")

# Load scaler (not the old model)
scaler = joblib.load("scaler.pkl")

# Sidebar Info
with st.sidebar:
    st.title("🧠 About Project")
    st.markdown("""
    This app detects anomalies (fraudulent transactions) using:

    - **DBSCAN Clustering** (Unsupervised ML)
    - **Z-Score Scaling**
    - **PCA Visualization**

    📊 Ideal for financial transaction analysis.
    """)

# Main Title
st.title("💳 Anomaly Detection in Financial Transactions")
st.markdown("🔍 Upload a `.csv` file to detect fraudulent transactions using unsupervised learning.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your CSV file here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Preview of Uploaded Data")
    st.dataframe(df.head())

    if st.button("🚀 Detect Anomalies"):
        with st.spinner("🔍 Processing..."):

            # Select numeric features
            numeric_df = df.select_dtypes(include=np.number)
            scaled_data = scaler.transform(numeric_df)

            # Apply DBSCAN with tuned parameters
            dbscan = DBSCAN(eps=2.2, min_samples=5)  # ✅ Use your tuned parameters
            labels = dbscan.fit_predict(scaled_data)

            # Label data
            df['Anomaly_Label'] = labels
            df['Anomaly_Status'] = df['Anomaly_Label'].apply(lambda x: 'Fraud' if x == -1 else 'Normal')

            # PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

            # Summary
            total = len(df)
            frauds = (df['Anomaly_Label'] == -1).sum()
            fraud_percent = frauds / total * 100

            st.success("✅ Anomaly detection complete!")

            st.subheader("📊 Summary Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total)
            col2.metric("Detected Frauds", frauds)
            col3.metric("Fraud Rate", f"{fraud_percent:.2f}%")

            

            # PCA Scatter Plot
            st.subheader("🧬 PCA Cluster Visualization")
            fig = px.scatter(
                df,
                x="PCA1",
                y="PCA2",
                color="Anomaly_Status",
                size="Amount" if "Amount" in df.columns else None,
                hover_data=["Amount"] if "Amount" in df.columns else None,
                color_discrete_map={"Normal": "green", "Fraud": "red"},
                title="Detected Clusters (2D PCA)"
            )
            st.plotly_chart(fig)

            # Pie Chart
            st.subheader("📈 Fraud vs Normal Distribution")
            pie = px.pie(df, names="Anomaly_Status", title="Detected Transaction Types")
            st.plotly_chart(pie)

            # Amount Boxplot
            if "Amount" in df.columns:
                st.subheader("💰 Transaction Amount Distribution")
                box = px.box(df, x="Anomaly_Status", y="Amount", color="Anomaly_Status",
                             color_discrete_map={"Normal": "green", "Fraud": "red"})
                st.plotly_chart(box)

            # Fraud filter
            if st.checkbox("🔎 Show only frauds"):
                st.dataframe(df[df['Anomaly_Status'] == 'Fraud'])

            # Download button
            result_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Labeled CSV", result_csv, file_name="anomaly_results.csv", mime="text/csv")
