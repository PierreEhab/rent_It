# Linear Regression Code Here
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from PreProcessing import preprocessing
from helper_functions import write_to_csv, normalize_data
import seaborn as sn


train_data, test_data = preprocessing()

test_data = test_data.drop(['ID'], axis=1)
# test_data = test_data.drop(['Dew point temperature(c)'], axis=1)
# test_data = test_data.drop(['ID'], axis=1)
y = train_data['Rented Bike Count'].values
x = train_data.drop(['Rented Bike Count'], axis=1).values
x = normalize_data(x)
test_data = normalize_data(test_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
linearRegressor = LinearRegression()
linearRegressor.fit(x_train, y_train)
test_data = np.array(test_data)
y_predicted = linearRegressor.predict(test_data)

# write to csv file
write_to_csv('predictedFromLR.csv', y_predicted)

