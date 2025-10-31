import streamlit as st
import requests

API_URL="http://127.0.0.1:8000/predict"

st.title("Diabetes Risk Prediction App (Females ≥ 21 years)")

st.markdown("This app uses an ML model served by FastAPI to predict **diabetes risk**.")

# Required fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
weight = st.number_input("Weight (kg)", min_value=1.0, step=0.1)
height = st.number_input("Height (m)", min_value=0.1, step=0.01, format="%.2f")
age = st.number_input("Age (years)", min_value=21, step=1)

# Optional fields (with units/type specified)
glucose = st.number_input("Glucose level (mg/dL, optional)", min_value=0.0, step=0.1)
blood_pressure = st.number_input("Blood Pressure (Diastolic, mm Hg, optional)", min_value=0.0, step=0.1)
skin_thickness = st.number_input("Skin Thickness (mm, optional)", min_value=0.0, step=0.1)
insulin = st.number_input("Insulin (µU/ml, optional)", min_value=0.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function (optional)", min_value=0.0, step=0.01)

# Button to send request
if st.button("Predict"):
    # Prepare JSON payload as FastAPI expects (same as schemas.DiabetesInput)
    payload = {
        "Pregnancies": pregnancies,
        "Weight": weight,
        "Height": height,
        "Age": age,
        "Glucose": glucose if glucose > 0 else None,
        "BloodPressure": blood_pressure if blood_pressure > 0 else None,
        "SkinThickness": skin_thickness if skin_thickness > 0 else None,
        "Insulin": insulin if insulin > 0 else None,
        "DiabetesPedigreeFunction": dpf if dpf > 0 else None,
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            if result["prediction"] == 1:
                # Converts the JSON response from FastAPI into a Python dictionary
                st.error(f"High risk of diabetes (probability: {result['probability']})")
            else:
                st.success(f"Low risk of diabetes (probability: {result['probability']})")
            #st.json(result)

        else:
            st.error(f"Error {response.status_code}: {response.text}")
    
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI backend. Make sure it is running.")

# More Info Section
st.markdown("---")
st.subheader("More Info")
st.markdown("""
The training dataset for the machine learning model powering this app is originally from the **National Institute of Diabetes and Digestive and Kidney Diseases**.  
The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.  

Several constraints were placed on the selection of these instances from a larger database.  
In particular, all patients here are **females at least 21 years old of Pima Indian heritage**.   

**Disclaimer:**  
This app is for **educational and informational purposes only**.  
It uses machine learning predictions based on the Pima Indians Diabetes Dataset and **is not a substitute for professional medical advice, diagnosis, or treatment**.  

Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.
👉 [Read more about the dataset here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
""")
