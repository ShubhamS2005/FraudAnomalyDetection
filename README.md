# 💳 Anomaly Detection in Financial Transactions  
### 🧠 A Machine Learning Approach using DBSCAN Clustering & PCA Visualization

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/ML-DBSCAN-yellowgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Unsupervised-Learning-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square" />
</div>

---

## 📌 Project Overview

This project identifies fraudulent transactions using **unsupervised clustering** with **DBSCAN**.  
Without needing labeled data, it detects frauds based on **density anomalies**, and evaluates them against real labels.

Visual inspection using **PCA** and metrics like **Silhouette Score** and **Confusion Matrix** validate the model.

---

## 🧠 Key Features

- 📈 DBSCAN Clustering (Unsupervised Anomaly Detection)
- 🧮 Z-score Standardization for feature scaling
- 🧠 PCA Visualization (2D) of fraud and normal clusters
- 📊 Silhouette Score + Confusion Matrix + Classification Report
- 💾 Saved Model & Scaler for deployment use

---

## 📁 Dataset Info

- 📍 Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 📦 Total Records: 284,807
- ⚠️ Class Distribution: 0 = Normal, 1 = Fraud (~0.17%)
- Features: 28 anonymized (V1–V28) + Time + Amount

---

---

## ⚙️ Libraries & Tools Used

| Category        | Tools & Libraries                           |
|----------------|----------------------------------------------|
| Programming    | Python 3.8+                                  |
| Data Handling  | pandas, numpy                                |
| ML Algorithms  | scikit-learn (DBSCAN, PCA, StandardScaler)   |
| Evaluation     | sklearn.metrics (confusion matrix, F1 score) |
| Visualization  | matplotlib, seaborn                          |
| Model Saving   | joblib                                       |

---

## 🧪 Evaluation Metrics

- ✅ **Silhouette Score**: Measures cluster compactness and separation.
- 📉 **Confusion Matrix**: Evaluates predictions vs actual frauds.
- 📋 **Classification Report**: Includes precision, recall, and F1-score.

> 📍 High **precision**: Most predicted anomalies were actual frauds.  
> ⚠️ Moderate **recall**: Common in unsupervised anomaly detection, as some frauds may not appear as clear outliers.

---

## How to run this Project

### Clone the repository
git clone https://github.com/ShubhamS2005/FraudAnomalyDetection.git
cd anomaly-detection-dbscan

### (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Run the Jupyter Notebook
jupyter notebook anomaly_detection.ipynb

---

## Project Structure

anomaly-detection-dbscan/
├── anomaly_detection.ipynb     ← Main project notebook
├── dbscan_model.pkl            ← Saved model for future use
├── scaler.pkl                  ← StandardScaler used for preprocessing
├── requirements.txt            ← Dependencies
├── plots/                      ← PCA & evaluation images
└── README.md                   ← Project documentation



