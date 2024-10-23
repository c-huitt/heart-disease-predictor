import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load('models/rank_1_xgboost_model_final.joblib')


# Function to preprocess user input
def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])

    input_df = input_df[['ca', 'oldpeak', 'age', 'chol', 'thalch', 'thal', 'trestbps']]

    scaler = model.named_steps['scaler']
    scaled_input = scaler.transform(input_df)

    return scaled_input


st.title('Heart Disease Predictor')

# Input fields for features
ca = st.number_input('Number of major vessels colored by fluoroscopy', min_value=0, max_value=4, value=0)
oldpeak = st.number_input('ST depression induced by exercise relative to rest', min_value=0.0, max_value=10.0,
                          value=0.0)
age = st.number_input('Age', min_value=0, max_value=120, value=30)
chol = st.number_input('Serum cholesterol in mg/dl', min_value=0, max_value=600, value=200)
thalch = st.number_input('Maximum heart rate achieved', min_value=0, max_value=250, value=150)
thal = st.selectbox('Thalassemia', options=[3, 6, 7],
                    format_func=lambda x: {3: 'Normal', 6: 'Fixed defect', 7: 'Reversible defect'}[x])
trestbps = st.number_input('Resting blood pressure (in mm Hg)', min_value=0, max_value=250, value=120)

# Creates a dictionary with user inputs
user_input = {
    'ca': ca,
    'oldpeak': oldpeak,
    'age': age,
    'chol': chol,
    'thalch': thalch,
    'thal': thal,
    'trestbps': trestbps
}

# Adds a prediction button
if st.button('Predict Heart Disease Risk'):
    # Preprocess the input
    processed_input = preprocess_input(user_input)

    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    # Display results
    st.subheader('Prediction Results:')
    if prediction[0] == 0:
        st.write('Low risk for heart disease')
    else:
        st.write('High risk for heart disease')


    # Confidence interval
    confidence = 0.95
    z_score = 1.96
    margin_of_error = z_score * np.sqrt((prediction_proba[0][1] * (1 - prediction_proba[0][1])) / len(processed_input))

    lower_bound = max(0, prediction_proba[0][1] - margin_of_error)
    upper_bound = min(1, prediction_proba[0][1] + margin_of_error)


st.sidebar.markdown("""
## About
This app predicts the risk of heart disease based on user input.
Please note that this is a simplified model and should not be used for medical diagnosis.
Always consult with a healthcare professional for medical advice.
""")