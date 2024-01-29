# -*- coding: utf-8 -*-
"""Heart Disease prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13nTRUyr5zQXrV8VR-jVgS2jHkjibORj_
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("heart.csv")

#dataset

#dataset["target"].describe()

y = dataset["target"]


target_temp = dataset.target.value_counts()

#print(target_temp)

plt.figure(figsize= (16,9))
sns.heatmap(dataset.corr(), annot = True, cmap='coolwarm', linewidths = 2)

dataset["sex"].unique()

dataset['sex'].value_counts().plot(kind='bar')

from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

"""Method 1 - Logistic Regression"""

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

Y_pred_lr = lr.predict(x_test)

print(classification_report(y_test, Y_pred_lr))

Y_pred_lr.shape

score_lr = round(accuracy_score(Y_pred_lr,y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

"""Method 2 - KNearestNeighbor(KNN)"""

from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier() # get instance of model
model2.fit(x_train, y_train) # Train/Fit model

y_pred2 = model2.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred2))

score_knn = round(accuracy_score(y_pred2,y_test)*100,2)

print("The accuracy score achieved using KNeighborsClassifier is: "+str(score_knn)+" %")

"""Method 3 - SVC"""

from sklearn.svm import SVC

model3 = SVC(random_state=1) # get instance of model
model3.fit(x_train, y_train) # Train/Fit model

y_pred3 = model3.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred3)) # output accuracy

score_svc = round(accuracy_score(y_pred3,y_test)*100,2)

print("The accuracy score achieved using SVC is: "+str(score_svc)+" %")

"""Method 4 - Naive Bayes"""

from sklearn.naive_bayes import GaussianNB

model4 = GaussianNB() # get instance of model
model4.fit(x_train, y_train) # Train/Fit model

y_pred4 = model4.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred4)) # output accuracy

score_nb = round(accuracy_score(y_pred4,y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

"""Method 5 - decision tree"""

from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(random_state=1) # get instance of model
model5.fit(x_train, y_train) # Train/Fit model

y_pred5 = model5.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred5))

score_dt = round(accuracy_score(y_pred5,y_test)*100,2)

print("The accuracy score achieved using DecisionTreeClassifier is: "+str(score_dt)+" %")

"""Method 6 - Random Forest Algorithm"""

from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(random_state=1)# get instance of model
model6.fit(x_train, y_train) # Train/Fit model

y_pred6 = model6.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred6)) # output accuracy

score_rf = round(accuracy_score(y_pred6,y_test)*100,2)

print("The accuracy score achieved using RandomForestClassifier is: "+str(score_rf)+" %")


# input feature values
input_data = (58,0,3,150,283,1,0,162,0,1,2,0,2)

# changing data to numpy array
input_data_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped =  input_data_array.reshape(1,-1)

# standarize the input data
# std_data = scaler.transform(input_data_reshaped)
# print(std_data[0])

#pip install streamlit --quiet

import streamlit
import pickle
import joblib
# importing the library

filename = "trained_model.pkl"
pickle.dump(model6, open(filename, 'wb'))
# saving file

loaded_model = pickle.load(open("trained_model.pkl",'rb'))

filename = 'heart_model.sav'
joblib.dump(model6, filename)
%%writefile app.py
import streamlit as st
import joblib

import numpy as np
import pickle

loaded_model = pickle.load(open('trained_model.pkl','rb'))
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

    if st.button('Diagnosis Test Result'):
        predict = Heart_disease_Prediction([age, sex, cp, restbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal ])

    st.success(predict)

    st.markdown("***")


if __name__ == '__main__':
    main()







