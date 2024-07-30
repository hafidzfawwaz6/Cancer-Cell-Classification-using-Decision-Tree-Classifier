# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model with the absolute path
model_path = 'decision_tree_model.pkl'
model = joblib.load(model_path)

# Define the input fields for the features
def user_input_features():
    sample_code_number = st.text_input('Sample Code Number', value='0')    
    clump_thickness = st.number_input('Clump Thickness', min_value=1, max_value=10, value=5)
    uniformity_of_cell_size = st.number_input('Uniformity of Cell Size', min_value=1, max_value=10, value=5)
    uniformity_of_cell_shape = st.number_input('Uniformity of Cell Shape', min_value=1, max_value=10, value=5)
    marginal_adhesion = st.number_input('Marginal Adhesion', min_value=1, max_value=10, value=5)
    single_epithelial_cell_size = st.number_input('Single Epithelial Cell Size', min_value=1, max_value=10, value=5)
    bare_nuclei = st.number_input('Bare Nuclei', min_value=1, max_value=10, value=5)
    bland_chromatin = st.number_input('Bland Chromatin', min_value=1, max_value=10, value=5)
    normal_nucleoli = st.number_input('Normal Nucleoli', min_value=1, max_value=10, value=5)
    mitoses = st.number_input('Mitoses', min_value=1, max_value=10, value=5)
    
    data = [
        int(sample_code_number),
        int(clump_thickness),
        int(uniformity_of_cell_size),
        int(uniformity_of_cell_shape),
        int(marginal_adhesion),
        int(single_epithelial_cell_size),
        int(bare_nuclei),
        int(bland_chromatin),
        int(normal_nucleoli),
        int(mitoses)
    ]

    sc = StandardScaler()
    return sc.transform([data])
    

# Title of the app
st.title('Cancer Cell Prediction App')

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input features')
st.write(input_df)

# Add a button to make the prediction
if st.button('Predict'):
    # Predict the class using the model
    prediction = model.predict(input_df)

    if prediction[0] == 1: 
        st.success('Class 4 (Malignant)')
    else: 
        st.error('Class 2 (Benign)') 



