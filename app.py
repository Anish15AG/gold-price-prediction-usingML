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

  