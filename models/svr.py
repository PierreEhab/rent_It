# SVR Code Here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from PreProcessing import preprocessing
from helper_functions import write_to_csv, normalize_data

train_data, test_data = preprocessing()
test_data = test_data.drop(['ID'], axis=1)
y = train_data['Rented Bike Count'].values
X = train_data.drop(['Rented Bike Count'], axis=1).values
X = normalize_data(X)
test_data = normalize_data(test_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# create the model object
regressor = SVR(kernel='linear')
regressor.fit(X_train, y_train)

y_predtrain = regressor.predict(X_test)
y_pred = regressor.predict(test_data)
write_to_csv('PredictedFromSVR.csv', y_pred)

# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_predtrain)), '.3f'))
print("\nRMSE: ", rmse)
