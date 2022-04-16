# ANN code Here
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import layers, models
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from PreProcessing import preprocessing
from tensorflow.keras.models import Sequential
import numpy as np
from helper_functions import write_to_csv

seed = 23
train_data, test_data = preprocessing()

# Get X and Y

x = train_data.drop(['Rented Bike Count'], axis=1).values
y = train_data['Rented Bike Count'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', )
model.summary()

history = model.fit(x_train, y_train, epochs=100, validation_split=0.2, validation_data=(x_test, y_test))

# Predict on test data
print(test_data.shape)
predictions = model.predict(test_data.drop(['ID'], axis=1))
# Save to csv file
write_to_csv('predictedFromANN.csv', predictions)
