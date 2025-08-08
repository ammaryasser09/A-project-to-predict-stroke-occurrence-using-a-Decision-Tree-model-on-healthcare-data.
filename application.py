import streamlit as st
import numpy as np
import onnxruntime as ort

@st.cache_resource
def load_model():
    return ort.InferenceSession('C:/Users/User/decision_tree_onnx_app2/stroke_prediction.onnx')

session = load_model()


st.title("Stroke Prediction API")
st.write("This API predicts the risk of stroke based on patient medical data such as age, gender, blood pressure, glucose level, BMI, and lifestyle factors. The prediction is powered by a machine learning model trained using decision trees on real-world health data.\nInput: Patient information (age, gender, hypertension, heart disease, BMI, etc.).\nOutput: Stroke risk (0 = No risk, 1 = At risk)\nNote: This tool is for educational/demo purposes and not intended for medical diagnosis.")

input_str = st.text_input("Enter patient data as comma-separated values (e.g., 56,1,0,0,90.0,26.4,0):")

if st.button("Predict"):
    try:
        input_list = list(map(float, input_str.split(',')))
        
        input_shape = session.get_inputs()[0].shape
        expected_features = input_shape[1]
        
        if len(input_list) != expected_features:
            st.write(f"Expected {expected_features} features but got {len(input_list)}")
        else:
            input_array = np.array([input_list], dtype=np.float32)
            input_name = session.get_inputs()[0].name
            pred = session.run(None, {input_name: input_array})
            
         
            prediction = pred[0][0]
            
            st.write(f"Prediction: {prediction}")  
    except Exception as e:
        st.write(f"Invalid input: {e}")
