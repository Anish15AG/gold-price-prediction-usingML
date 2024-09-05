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

#Conversion of all columns to Numeric values to avoid errors during correlation
gold_data = gold_data.apply(pd.to_numeric, errors='coerce')

#calculation of correlation matrix
correlation = gold_data.corr()

#constructing a heatmap to understand the correlation between the values
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

#correlation values of gold
print(correlation['GLD'])

#understanding the distribution of the GLD using distplot
sns.distplot(gold_data['GLD'],color='green')