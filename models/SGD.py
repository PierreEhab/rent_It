import numpy as np
import pandas as pd
from sklearn import cross_validation, grid_search, linear_model, metrics, pipeline, preprocessing

# Date Reading
train_data = pd.read_csv('data set/train.csv')
test_data = pd.read_csv('data set/test.csv')

train_labels = train_data['Rented Bike Count'].values
train_data = train_data.drop(['Date', 'Rented Bike Count', 'Temperature(°C)'], axis = 1)
test_labels = test_data['Rented Bike Count'].values
test_data = test_data.drop(['Date', 'Rented Bike Count', 'Temperature(°C)'], axis = 1)

binary_data_columns = ['Holiday', 'Functioningday']
binary_data_indices = np.array([(column in binary_data_columns) for column in train_data.columns], dtype = bool)

categorical_data_columns = ['season', 'weather', 'month']
categorical_data_indices = np.array([(column in categorical_data_columns) for column in train_data.columns], dtype = bool)

numeric_data_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'hour']
numeric_data_indices = np.array([(column in numeric_data_columns) for column in train_data.columns], dtype = bool)