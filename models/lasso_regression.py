from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  mean_absolute_error
from PreProcessing import preprocessing
from helper_functions import write_to_csv, normalize_data, standardize_data

lasso = Lasso()
train_data, test_data = preprocessing()
test_data = test_data.drop(['ID'], axis=1)
y = train_data['Rented Bike Count'].values.reshape(-1, 1)
x = train_data.drop(['Rented Bike Count'], axis=1).values
# Standardization
x = standardize_data(x)

x = normalize_data(x)
test_data = normalize_data(test_data)
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

lasso_regressor.fit(x, y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

prediction_lasso = lasso_regressor.predict(test_data)

write_to_csv('predictedFromLassoReg.csv', prediction_lasso)

y_pred = lasso_regressor.predict(x_test)
print("MAE",mean_absolute_error(y_test,y_pred))
