import streamlit as st
import pandas as pd
import joblib

model = joblib.load('stroke_classifier.joblib')

def predict(model, input_df):
    predictions = model.predict(input_df)
    return ['high likelihood' if prediction == 1 else 'lower likelihood' for prediction in predictions]

def process_input(input_df):
    input_df.loc[input_df.age < 17, 'work_type'] = 'children'
    input_df = input_df[input_df.work_type != 'Never_worked']
    input_df['smoking_status'] = input_df['smoking_status'].replace({'Unknown': 'no_answer'})
    cat_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    input_df[cat_features] = input_df[cat_features].astype('category')
    return input_df

st.title("Stroke Prediction App")

gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=20.0)
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'no_answer'])

input_df = pd.DataFrame({'gender': [gender], 'age': [age],
                         'hypertension': [hypertension],
                         'heart_disease': [heart_disease],
                         'ever_married': [ever_married],
                         'work_type': [work_type],
                         'Residence_type': [Residence_type],
                         'avg_glucose_level': [avg_glucose_level],
                         'bmi': [bmi],
                         'smoking_status': [smoking_status]})

processed_input_df = process_input(input_df)

if st.button('Predict'):
    prediction = predict(model, processed_input_df)
    st.write('Based on the inputs, the predicted likelihood of having a stroke is: ', prediction)