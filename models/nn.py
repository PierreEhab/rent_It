# ANN code Here
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import layers, models
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from PreProcessing import preprocessing
from tensorflow.keras.models import Sequential
import numpy as np
from helper_functions import write_to_csv

seed = 23
train_data, test_data = preprocessing()
test_data = test_data.drop(['ID'], axis=1)
# Get X and Y

x = train_data.drop(['Rented Bike Count'], axis=1).values
y = train_data['Rented Bike Count'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaler_fit = scaler.fit(x)
x = scaler_fit.transform(x)
test_scaler = MinMaxScaler()
test_scaler_fit = test_scaler.fit(test_data)
test_data = test_scaler_fit.transform(test_data)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)


model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', )
model.summary()

history = model.fit(x_train, y_train, epochs=100, validation_split=0.2, validation_data=(x_test, y_test))

# Predict on test data
predictions = model.predict(test_data)
# Save to csv file
write_to_csv('predictedFromNN.csv', predictions)
