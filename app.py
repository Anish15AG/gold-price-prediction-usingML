'''
This file is only for reference to host it on Streamlit/LocalHost
'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


#Load the DataSet
st.title("Gold Prediction Application")

#Upload the DataSet
uploaded_file = st.file_uploader('/content/drive/MyDrive/GoldPricePrediction_dataset/gld_price_data.csv', type="csv")

if uploaded_file is not None:
  gold_data = pd.read_csv(uploaded_file)

  #Display the DataSet
  st.write("DataSet Preview: ")
  st.DataFrame(gold_data.head())

  #Data Cleaning and Processing
  gold_data = gold_data.apply(pd.to_numeric, errors='coerce')

  #Creation of Feature Set
  X = gold_data.drop(['Date', 'GLD'], axis=1)
  Y = gold_data['GLD']

  #Splitting the data into Training and Test Data respectively
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

  #Training the Random Forest Model
  regressor = RandomForestRegressor(n_estimators=100)
  regressor.fit(X_train, Y_train)


  #Predictions on the Test Data
  test_data_prediction = regressor.predict(X_test)

  #R Squared Error method to check the accuracy of the model
  error_score = metrics.r2_score(Y_test, test_data_prediction)
  st.write(f"R Squared Error: {error_score:.2f}")

  #Plotting the Actual and the Predicted Values
  fig, ax = plt.subplots()
  plt.plot(list(Y_test), color='blue', label='Actual Value')
  plt.plot(test_data_prediction, color='green', label='Predicted Values')
  plt.title('Actual vs Predicted Values')
  plt.xlabel('Number of Values')
  plt.ylabel('Gold Price in $(USD)')
  plt.legend()
  st.pyplot(fig)