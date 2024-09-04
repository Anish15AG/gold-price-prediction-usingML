# -*- coding: utf-8 -*-
"""
GoldPricePrediction_usingML.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

"""Importing the DataSet using the read_csv function of the pandas library"""

#loading the dataset to a Pandas DataFrame
gold_data = pd.read_csv('/content/gld_price_data.csv')

#print the first 5 rows to understand the initial timeframe of the DataSet
gold_data.head()

#printing the last 5 rows to understand the dataset's last updated date
gold_data.tail()

#number of rows and columns present in the dataset
gold_data.shape

#checking if there are any null values or imputation is required/cleaning of data
gold_data.info()

#checking for any missing values
gold_data.isnull().sum()

#statistical measures of the dataset
gold_data.describe()