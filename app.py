# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model with the absolute path
model_path = 'decision_tree_model.pkl'
model = joblib.load(model_path)
sc = StandardScaler()

# Define the input fields for the features
def user_input_data():
    sample_code_number = st.text_input('Sample Code Number', value='1018561')    
    clump_thickness = st.number_input('Clump Thickness', min_value=1, max_value=10, value=2)
    uniformity_of_cell_size = st.number_input('Uniformity of Cell Size', min_value=1, max_value=10, value=1)
    uniformity_of_cell_shape = st.number_input('Uniformity of Cell Shape', min_value=1, max_value=10, value=2)
    marginal_adhesion = st.number_input('Marginal Adhesion', min_value=1, max_value=10, value=1)
    single_epithelial_cell_size = st.number_input('Single Epithelial Cell Size', min_value=1, max_value=10, value=2)
    bare_nuclei = st.number_input('Bare Nuclei', min_value=1, max_value=10, value=1)
    bland_chromatin = st.number_input('Bland Chromatin', min_value=1, max_value=10, value=3)
    normal_nucleoli = st.number_input('Normal Nucleoli', min_value=1, max_value=10, value=1)
    mitoses = st.number_input('Mitoses', min_value=1, max_value=10, value=2)
    
    data = {
        'Sample Code Number': sample_code_number,
        'Clump Thickness': clump_thickness,
        'Uniformity of Cell Size': uniformity_of_cell_size,
        'Uniformity of Cell Shape': uniformity_of_cell_shape,
        'Marginal Adhesion': marginal_adhesion,
        'Single Epithelial Cell Size': single_epithelial_cell_size,
        'Bare Nuclei': bare_nuclei,
        'Bland Chromatin': bland_chromatin,
        'Normal Nucleoli': normal_nucleoli,
        'Mitoses': mitoses
    }
    return data    

# Title of the app
st.title('Cancer Cell Prediction App')

# Get user input
data = user_input_data()

# Display user input
st.subheader('User Input features')
st.write(pd.DataFrame(data, index=[0]))

# Add a button to make the prediction
if st.button('Predict'):
    # Predict the class using the model
    data_array = []
    for key, value in data.items():
        data_array.append(int(value))
    
    sc.transform(data_array)
    prediction = model.predict(sc.transform([data_array]))

    if prediction[0] == 1: 
        st.success('Class 4 (Malignant)')
    else: 
        st.error('Class 2 (Benign)') 