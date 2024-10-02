import streamlit as st
import pickle
import pandas as pd

pipe = pickle.load(open('xgboost_pipeline.pkl', 'rb'))

st.title('Software Developer Salary Predictor')

age = ['25-34 years old', '35-44 years old', '45-54 years old',
       '55-64 years old', '18-24 years old', '65 years or older',
       'Under 18 years old']

country = ['Austria', 'Other',
           'United Kingdom of Great Britain and Northern Ireland',
           'United States of America', 'France', 'Germany', 'Brazil',
           'Ukraine', 'Canada', 'Italy', 'Switzerland', 'India', 'Spain',
           'Netherlands', 'Sweden', 'Poland', 'Australia']

education_level = ['Professional degree', 'Master’s degree', 'Less than a Bachelors',
                   'Bachelor’s degree']

col1, col2 = st.columns(2)

with col1:
    age_ = st.selectbox('Select your Age', sorted(age))
with col2:
    country_ = st.selectbox('Select your Country', sorted(country))

education_level_ = st.selectbox('Select your Education Level', sorted(education_level))

experience = st.slider("Years of Experience", 0, 50, 3)

# Predict button
if st.button('Predict Salary'):
    input_df = pd.DataFrame({
        'Country': [country_],
        'Age': [age_],
        'EdLevel': [education_level_],
        'YearsCodePro': [experience]
    })
    
    st.table(input_df)
    
    result = pipe.predict(input_df)
    
    st.subheader(f"The estimated salary is ${result[0]:,.2f}")
