import math

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from helper_functions import write_to_csv, normalize_data
from PreProcessing import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_data, test_data = preprocessing()
test_data = test_data.drop(['ID'], axis=1)
y = train_data['Rented Bike Count'].values.reshape(-1, 1)
x = train_data.drop(['Rented Bike Count'], axis=1).values
x = normalize_data(x)
test_data = normalize_data(test_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

reg = Ridge(alpha=0.005)
reg.fit(x_train, y_train)
y_pred = reg.predict(test_data)
write_to_csv('predictions/predictedFromRidgeReg.csv', y_pred)
y_pred = reg.predict(x_test)
print("MAE",mean_absolute_error(y_test,y_pred))
print("RMSE", math.sqrt(mean_squared_error(y_test, y_pred)))