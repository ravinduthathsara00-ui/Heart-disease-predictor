import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# --- Load the saved model and scaler ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Display debug info in sidebar
    n_features_scaler = scaler.n_features_in_
    st.sidebar.write(f"📊 Scaler expects: {n_features_scaler} features")

except FileNotFoundError:
    st.error("⚠️ Model files not found! Please make sure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# --- App Title ---
st.title("❤️ Heart Disease Risk Prediction System")
st.markdown("""
This tool uses a Machine Learning model to assess the risk of heart disease 
based on your symptoms and lifestyle factors. 
**Please fill in all the details below.** 
*(0 = No / Absent, 1 = Yes / Present)*
""")
st.divider()

# --- INPUT SECTION ---
st.subheader("🩺 Enter Patient Health Data")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Symptoms**")
    Chest_Pain = st.selectbox(
        "Chest Pain (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Shortness_of_Breath = st.selectbox(
        "Shortness of Breath (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Fatigue = st.selectbox(
        "Fatigue (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Palpitations = st.selectbox(
        "Palpitations (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Dizziness = st.selectbox(
        "Dizziness (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Swelling = st.selectbox(
        "Swelling (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Pain_Areas_Back = st.selectbox(
        "Pain in Back/Areas (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    # 🛑 CHANGED: Combined Cold Sweats and Nausea into ONE input
    Cold_Sweats_Nausea = st.selectbox(
        "Cold Sweats / Nausea (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col2:
    st.markdown("**Medical History & Lifestyle**")
    High_BP = st.selectbox(
        "High Blood Pressure (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    High_Cholesterol = st.selectbox(
        "High Cholesterol (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Diabetes = st.selectbox(
        "Diabetes (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Smoking = st.selectbox(
        "Smoking (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Obesity = st.selectbox(
        "Obesity (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Sedentary_Lifestyle = st.selectbox(
        "Sedentary Lifestyle (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Family_History = st.selectbox(
        "Family History of Heart Disease (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Overt_Dress = st.selectbox(
        "Overt Dress / Physical Strain (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    Gender = st.selectbox(
        "Gender (0=Female, 1=Male)",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )
    
    Age = st.slider("Age (years)", min_value=18, max_value=100, value=45, step=1)

st.divider()

# --- PREDICTION BUTTON ---
if st.button("🔮 Predict Heart Disease Risk", type="primary"):
    # 🛑 UPDATED: Now using EXACTLY 18 features (Cold_Sweats_Nausea is combined)
    # This matches the columns in your CSV:
    # Chest_Pain, Shortness_of_Breath, Fatigue, Palpitations, Dizziness, Swelling,
    # Pain_Areas_Back, Cold_Sweats_Nausea, High_BP, High_Cholesterol, Diabetes,
    # Smoking, Obesity, Sedentary_Lifestyle, Family_History, Overt_Dress, Gender, Age
    
    input_features = [
        Chest_Pain,
        Shortness_of_Breath,
        Fatigue,
        Palpitations,
        Dizziness,
        Swelling,
        Pain_Areas_Back,
        Cold_Sweats_Nausea,  # <-- Combined field
        High_BP,
        High_Cholesterol,
        Diabetes,
        Smoking,
        Obesity,
        Sedentary_Lifestyle,
        Family_History,
        Overt_Dress,
        Gender,
        Age
    ]
    
    # Debug check (this shows in the Streamlit logs)
    print(f"Number of input features: {len(input_features)}")
    print(f"Input values: {input_features}")
    
    # Convert to numpy array and reshape for the model
    input_array = np.array([input_features])
    
    try:
        # Scale the input using the scaler
        input_scaled = scaler.transform(input_array)
        
        # --- Make Prediction ---
        prediction = model.predict(input_scaled)[0]
        
        # Get probability (confidence score)
        probability = model.predict_proba(input_scaled)[0]
        
        # --- Display Results ---
        st.divider()
        st.subheader("📋 Prediction Result")
        
        if prediction == 1:
            st.error("⚠️ **HIGH RISK**")
            st.markdown(f"""
            The model predicts a **high risk** of heart disease.  
            **Confidence:** {probability[1]*100:.2f}%  
            Please consult a healthcare professional immediately.
            """)
        else:
            st.success("✅ **LOW RISK**")
            st.markdown(f"""
            The model predicts a **low risk** of heart disease.  
            **Confidence:** {probability[0]*100:.2f}%  
            Continue maintaining a healthy lifestyle!
            """)
        
        # Add a disclaimer
        st.caption("⚠️ Disclaimer: This is a machine learning prediction tool and is for educational purposes only. It should NOT replace professional medical diagnosis or advice. Always consult a qualified doctor.")
    
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
        st.info("Please make sure all inputs are filled correctly.")

# --- Debug Info in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.caption("Debug Info:")
try:
    st.sidebar.write(f"📊 Scaler expects: {scaler.n_features_in_} features")
    st.sidebar.write(f"📊 You are providing: 18 features")
except:
    pass
