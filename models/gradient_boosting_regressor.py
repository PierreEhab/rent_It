import math
import numpy as np
from sklearn.model_selection import train_test_split
from PreProcessing import preprocessing
from helper_functions import write_to_csv, normalize_data, standardize_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

train_data, test_data = preprocessing()

test_data = test_data.drop(['ID'], axis=1)
y = train_data['Rented Bike Count'].values
x = train_data.drop(['Rented Bike Count'], axis=1).values

x = normalize_data(x)
test_data = normalize_data(test_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=24)

model = GradientBoostingRegressor()
model.fit(x, y)
test_data = np.array(test_data)
y_predicted = model.predict(test_data)

# write to csv file
write_to_csv('predictions/predictedFromGradientBoostingRegressor4.csv', y_predicted)
print(y_predicted.shape)

model = GradientBoostingRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("MAE",mean_absolute_error(y_test,y_pred))
print("RMSE", math.sqrt(mean_squared_error(y_test, y_pred)))