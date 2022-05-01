import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from PreProcessing import preprocessing
from helper_functions import normalize_data, write_to_csv

train_data, test_data = preprocessing()

test_data = test_data.drop(['ID'], axis=1)

y = train_data['Rented Bike Count'].values
x = train_data.drop(['Rented Bike Count'], axis=1).values
x = normalize_data(x)
test_data = normalize_data(test_data)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)
randForest = RandomForestRegressor()
randForest.fit(x, y)
y_pred = randForest.predict(test_data)
print(y_pred.shape)
write_to_csv('predictions/predictedFromRandomForest.csv', y_pred)

'''y_pred = randForest.predict((x_test))
print("MAE", mean_absolute_error(y_test, y_pred))
print("RMSE", math.sqrt(mean_squared_error(y_test, y_pred)))'''
