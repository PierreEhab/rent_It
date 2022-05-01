import math

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import  mean_absolute_error,mean_squared_error
from PreProcessing import preprocessing
from helper_functions import write_to_csv, normalize_data, standardize_data

train_data, test_data = preprocessing()
test_data = test_data.drop(['ID'], axis=1)
y = train_data['Rented Bike Count'].values.reshape(-1, 1)
x = train_data.drop(['Rented Bike Count'], axis=1).values

x = normalize_data(x)
test_data = normalize_data(test_data)

treesReg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(x, y)
#treesReg.score(x_test, y_test)
prediction_treesReg = treesReg.predict(test_data)

write_to_csv('predictions/predictedFromTreesReg.csv', prediction_treesReg)
print(prediction_treesReg.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)
treesReg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(x_train, y_train)
treesReg.score(x_test, y_test)
y_pred = treesReg.predict(x_test)
print("MAE", mean_absolute_error(y_test, y_pred))
print("RMSE", math.sqrt(mean_squared_error(y_test, y_pred)))

