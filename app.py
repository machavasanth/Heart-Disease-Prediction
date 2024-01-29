import streamlit as st
import joblib

import numpy as np
import pickle


# loading the saved model


loaded_model = pickle.load(open('trained_model.pkl','rb'))

# creating a function for prediction

def Heart_disease_Prediction(input_data):


    # changing data to numpy array
    input_data_array = np.asarray(input_data, dtype = np.float64)

    # reshape the array as we are predicting for one instance
    input_data_reshaped =  input_data_array.reshape(1,-1)

    result = loaded_model.predict(input_data_reshaped)
    print("The prediction is : ",result)

    if (result[0] == 1):
      return "patient diagnosed with heart disease"
    else:
      return "patient is healthy and not diagnosed with heart disease"


def main():
    # giving a title
    st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease Prediction Application</h1>", unsafe_allow_html=True)

    # getting the input data from input user

    age= st.text_input("Age of person : ")
    sex= st.text_input("Sex (0-1): ")

    cp = st.text_input("Chest pain type (0-3): ")
    restbps = st.text_input("Resting BP : ")
    chol = st.text_input("Serum Cholestoral (mg/dl) : ")
    fbs = st.text_input("Fasting blood sugar > 120 mg/dl(1 = true,0 = false) : ")
    restecg = st.text_input("Resting electrocardiographic results (0-2) : ")

    thalach = st.text_input("Maximum heart rate achieved : ")
    exang = st.text_input("Exercise induced angina (0-1): ")
    oldpeak = st.text_input("Oldpeak : ")
    slope = st.text_input("Slope of the peak exercise ST segment (0-2): ")
    ca = st.text_input("Number of major vessels (0-3) : ")
    thal = st.text_input("chest pain type (0-3): ")


    # code for prediction
    predict = '' # null string

    # creating a button for prediction

    if st.button('Diagnose Test Result'):
        predict = Heart_disease_Prediction([age, sex, cp, restbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal ])

    st.success(predict)

    st.markdown("***")


if __name__ == '__main__':
    main()
