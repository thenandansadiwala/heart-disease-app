# ❤️ Heart Disease Diagnostic Pro
An End-to-End Machine Learning Pipeline for Cardiac Risk Assessment.

## 🚀 Overview
This project uses an **XGBoost Classifier** trained on over 600,000 clinical records to predict the probability of heart disease. It achieves a **0.954 AUC** on unseen test data.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Model:** XGBoost (Gradient Boosting)
* **Validation:** Stratified 5-Fold Cross-Validation
* **Interface:** Streamlit
* **Deployment:** Streamlit Community Cloud

## 📦 Installation & Local Run
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/heart-disease-app.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📊 Model Performance
* **CV ROC-AUC:** 0.9552
* **Kaggle Private Score:** 0.9549