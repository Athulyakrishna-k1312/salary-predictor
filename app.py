# app.py : Streamlit web app for employee salary classification

import streamlit as st
import pandas as pd
import joblib





# Load the trained model
model = joblib.load("best_model.pkl")

le_education = joblib.load("le_education.pkl")
le_occupation = joblib.load("le_occupation.pkl")
# Load others if needed (le_workclass, le_gender, etc.)

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Input fields
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 1)

# Create input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]

})

st.write("### Input Data")
st.write(input_df)

# Prediction

if st.button("Predict Salary Class"):
    try:
        # Encode the categorical columns
        input_df = input_df.drop('experience', axis=1)
        input_df['education'] = le_education.transform(input_df['education'])
        input_df['occupation'] = le_occupation.transform(input_df['occupation'])

        # Make prediction
        prediction = model.predict(input_df)

        st.success(f"🎯 Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")


# Batch prediction section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("🔍 Predictions:")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

