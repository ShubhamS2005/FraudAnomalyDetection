import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.express as px

# App Config
st.set_page_config(
    page_title="Anomaly Detection App",
    layout="wide",
    page_icon="💳"
)

# Detect dark mode
is_dark = st.get_option("theme.base") == "dark"

# Custom CSS
st.markdown(f"""
    <style>
    .main-title {{
        font-size: 42px;
        font-weight: bold;
        color: {'#58D68D' if is_dark else '#2c3e50'};
        margin-bottom: 10px;
    }}
    .subsection {{
        font-size: 24px;
        font-weight: bold;
        color: {'#F4D03F' if is_dark else '#1f618d'};
        margin-top: 25px;
    }}
    .metric-title {{
        font-size: 20px;
        color: {'#F1948A' if is_dark else '#117a65'};
    }}
    .block-container {{
        padding: 2rem 1rem 2rem 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🧠 About Project")
    st.success("""
    This app detects anomalies using:

    ✅ **DBSCAN Clustering**  
    📉 **Z-Score Scaling**  
    🧬 **PCA Visualization**

    Use it to analyze financial transactions and flag suspicious ones.
    """)

    st.markdown("## 🔧 DBSCAN Tuning")
    eps_val = st.slider("📏 Epsilon (eps)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
    min_samples_val = st.slider("👥 Min Samples", min_value=1, max_value=20, value=5, step=1)

# Main Title
st.markdown('<div class="main-title">💳 Anomaly Detection in Financial Transactions</div>', unsafe_allow_html=True)
st.markdown("🔍 **Upload a `.csv` file to begin fraud detection using unsupervised learning.**")

# File Upload
uploaded_file = st.file_uploader("📤 Upload your CSV file here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown('<div class="subsection">📌 Preview of Uploaded Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    if st.button("🚀 Detect Anomalies"):
        status_placeholder = st.empty()
        with st.spinner("🔍 Processing and clustering... Please wait"):
            numeric_df = df.select_dtypes(include=np.number)
            scaled_data = joblib.load("scaler.pkl").transform(numeric_df)

            # Apply DBSCAN using slider values
            dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
            labels = dbscan.fit_predict(scaled_data)

            df['Anomaly_Label'] = labels
            df['Anomaly_Status'] = df['Anomaly_Label'].apply(lambda x: 'Fraud' if x == -1 else 'Normal')

            pca_result = PCA(n_components=2).fit_transform(scaled_data)
            df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

            total = len(df)
            frauds = (df['Anomaly_Label'] == -1).sum()
            fraud_percent = frauds / total * 100

        status_placeholder.success("✅ Anomaly detection complete!")

        st.markdown('<div class="subsection">📊 Summary Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Total Transactions", total)
        col2.metric("🚨 Detected Frauds", frauds)
        col3.metric("📊 Fraud Rate", f"{fraud_percent:.2f}%")

        col4, col5 = st.columns(2)
        col4.metric("🔧 Epsilon (eps)", eps_val)
        col5.metric("🔧 Min Samples", min_samples_val)

        st.markdown('<div class="subsection">🧬 PCA Cluster Visualization</div>', unsafe_allow_html=True)
        fig = px.scatter(
            df,
            x="PCA1",
            y="PCA2",
            color="Anomaly_Status",
            size="Amount" if "Amount" in df.columns else None,
            hover_data=["Amount"] if "Amount" in df.columns else None,
            color_discrete_map={"Normal": "limegreen", "Fraud": "orangered"},
            template="plotly_dark" if is_dark else "plotly_white",
            title="Detected Clusters (2D PCA)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="subsection">📈 Fraud vs Normal Distribution</div>', unsafe_allow_html=True)
        pie = px.pie(df, names="Anomaly_Status", title="Detected Transaction Types",
                     color_discrete_map={"Normal": "limegreen", "Fraud": "orangered"},
                     template="plotly_dark" if is_dark else "plotly_white")
        st.plotly_chart(pie, use_container_width=True)

        if "Amount" in df.columns:
            st.markdown('<div class="subsection">💰 Transaction Amount Distribution</div>', unsafe_allow_html=True)
            box = px.box(df, x="Anomaly_Status", y="Amount", color="Anomaly_Status",
                         color_discrete_map={"Normal": "limegreen", "Fraud": "orangered"},
                         template="plotly_dark" if is_dark else "plotly_white")
            st.plotly_chart(box, use_container_width=True)

        if st.checkbox("🔎 Show only frauds"):
            st.markdown('<div class="subsection">⚠️ Detected Fraudulent Transactions</div>', unsafe_allow_html=True)
            st.dataframe(df[df['Anomaly_Status'] == 'Fraud'], use_container_width=True)

        st.markdown('<div class="subsection">📥 Download Results</div>', unsafe_allow_html=True)
        result_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Labeled CSV", result_csv, file_name="anomaly_results.csv", mime="text/csv")
