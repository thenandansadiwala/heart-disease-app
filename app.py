import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Heart Disease Diagnostic Pro",
    layout="centered"
)

# --- 2. THE BRAIN: LOADING THE PIPELINE ---
@st.cache_resource
def load_assets():
    """Load the model and column structure from the local pkl file."""
    model_path = os.path.join("model", "heart_model.pkl")
    if not os.path.exists(model_path):
        st.error(f"Error: {model_path} not found. Ensure you have run your training script first.")
        return None, None
    
    data = joblib.load(model_path)
    return data['model'], data['columns']

model, required_columns = load_assets()

# --- 3. UI COMPONENTS ---
st.title("Heart Disease Risk Predictor:", help="This tool uses an **XGBoost machine learning model** to estimate the probability of heart disease based on clinical vitals and test results.")

with st.form("diagnostic_form"):
    st.subheader("Patient Clinical Data")
    
    # Organize inputs into two columns for a clean UI
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", 1, 120, 50)
        sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        cp = st.selectbox("Chest Pain Type", options=[(1, "Typical Angina"), (2, "Atypical Angina"), 
                                                     (3, "Non-Anginal"), (4, "Asymptomatic")], 
                          format_func=lambda x: x[1])[0]
        bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

    with col2:
        max_hr = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
        st_dep = st.number_input("ST Depression (EKG)", 0.0, 10.0, 0.0, step=0.1)
        vessels = st.slider("Major Vessels (Fluoroscopy)", 0, 3, 0)
        thallium = st.selectbox("Thallium Test Result", options=[(3, "Normal"), (6, "Fixed Defect"), 
                                                               (7, "Reversible Defect")], 
                               format_func=lambda x: x[1])[0]

    # These features were in our data but are less critical for input
    # We set them to standard/mean defaults to avoid overwhelming the user
    fbs = 0 # Fasting Blood Sugar
    ekg = 0 # Resting EKG
    slope = 1 # Slope of ST
    
    submit = st.form_submit_button("Generate Diagnostic Report")

# --- 4. PREDICTION LOGIC ---
if submit and model is not None:
    # A. Create a Raw DataFrame from user inputs
    raw_input = pd.DataFrame([{
        'Age': age, 'Sex': sex, 'Chest pain type': cp, 'BP': bp,
        'Cholesterol': chol, 'FBS over 120': fbs, 'EKG results': ekg,
        'Max HR': max_hr, 'Exercise angina': exang, 'ST depression': st_dep,
        'Slope of ST': slope, 'Number of vessels fluro': vessels, 'Thallium': thallium
    }])

    # B. Apply One-Hot Encoding (The same logic used in training)
    categorical_cols = ['Chest pain type', 'EKG results', 'Slope of ST', 'Thallium']
    processed_input = pd.get_dummies(raw_input, columns=categorical_cols)

    # C. Alignment: Ensure the 18 columns match the model's brain exactly
    for col in required_columns:
        if col not in processed_input.columns:
            processed_input[col] = 0
            
    # Reorder columns to match the training order
    processed_input = processed_input[required_columns]

    # D. Final Prediction
    probability = model.predict_proba(processed_input)[:, 1][0]
    
    # --- 5. RESULTS DISPLAY ---
    st.divider()
    risk_color = "red" if probability > 0.5 else "green"
    
    st.markdown(f"<h2 style='text-align: center; color: {risk_color};'>Risk Score: {probability*100:.1f}%</h2>", 
                unsafe_allow_html=True)
    
    if probability > 0.5:
        st.warning("🚨 **High Probability Detected:** Please consult a cardiologist immediately for further testing.")
    else:
        st.success("✅ **Low Probability Detected:** Patient shows healthy clinical indicators.")

    # Show a brief summary for the doctor
    with st.expander("View Technical Data (SHAP Equivalent)"):
        st.write("Model utilized: XGBoost Classifier (Ensemble)")
        st.write(f"Input Features Processed: {len(required_columns)}")
        st.dataframe(processed_input)