# Necessary imports
import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

from PreProcessing import preprocessing
from helper_functions import normalize_data

train_data, test_data = preprocessing()

test_data = test_data.drop(['ID'], axis=1)
# Get X and Y

x = train_data.drop(['Rented Bike Count'], axis=1).values
y = train_data['Rented Bike Count'].values.reshape(-1, 1)


x = normalize_data(x)
test_data = normalize_data(test_data)
# Splitting
train_X, test_X, train_y, test_y = train_test_split(x, y,
                                                    test_size=0.3, random_state=123)

# Instantiation
xgb_r = xg.XGBRegressor(objective='reg:linear',n_estimators=10, seed=23)

# Fitting the model
xgb_r.fit(train_X, train_y)

# Predict the model
pred = xgb_r.predict(test_X)

# RMSE Computation
rmse = np.sqrt(MSE(test_y, pred))
print("RMSE : % f" % (rmse))