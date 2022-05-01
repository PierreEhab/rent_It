from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from PreProcessing import preprocessing
from helper_functions import normalize_data, write_to_csv

seed = 23
train_data, test_data = preprocessing()
test_data = test_data.drop(['ID'], axis=1)
y = train_data['Rented Bike Count'].values.reshape(-1, 1)
x = train_data.drop(['Rented Bike Count'], axis=1).values
x = normalize_data(x)
test_data = normalize_data(test_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)


regressor = DecisionTreeRegressor(random_state=seed)

# fit the regressor with X and Y data
regressor.fit(x, y)

predictions = regressor.predict(test_data)

write_to_csv('predictedFromDT.csv', predictions)

# regressor.fit(x_train, y_train)
#
# predictions = regressor.predict(x_test)
# print("MAE", mean_absolute_error(y_test, predictions))
# print("RMSE", math.sqrt(mean_squared_error(y_test, predictions)))

