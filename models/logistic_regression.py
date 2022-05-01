# Logistic Regression Code Here
import math

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from helper_functions import write_to_csv, normalize_data
from PreProcessing import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error

train_data, test_data = preprocessing()
test_data = test_data.drop(['ID'], axis=1)
y = train_data['Rented Bike Count'].values.reshape(-1, 1)
x = train_data.drop(['Rented Bike Count'], axis=1).values
x = normalize_data(x)
test_data = normalize_data(test_data)

logReg = LogisticRegression()
logReg.fit(x,y)
y_pred = logReg.predict(test_data)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)
logReg.fit(x_train,y_train)
y_pred = logReg.predict(x_test)
print("MAE",mean_absolute_error(y_test,y_pred))
print("MSE", mean_squared_error(y_test, y_pred))
print("RMSE", math.sqrt(mean_squared_error(y_test, y_pred)))