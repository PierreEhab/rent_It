# voting regressor Code Here
from PreProcessing import preprocessing
import math
from helper_functions import write_to_csv, normalize_data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn




train_data, test_data = preprocessing()

test_data = test_data.drop(['ID'], axis=1)

y = train_data['Rented Bike Count'].values
x = train_data.drop(['Rented Bike Count'], axis=1).values
x = normalize_data(x)
test_data = normalize_data(test_data)

#test_data = np.array(test_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=24)

#model = GradientBoostingRegressor()
#model.fit(x_train,y_train)
#test_data = np.array(test_data)
#y_predicted = model.predict(test_data)


reg1 = GradientBoostingRegressor()
reg2 = RandomForestRegressor()
reg3 = LinearRegression()

reg1.fit(x_train, y_train)
reg2.fit(x_train, y_train)
reg3.fit(x_train, y_train)
model= VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])

model.fit(x_train,y_train)

y_pred= model.predict(x_test)

#r = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
#r2_score = VotingRegressor.score(x_test,y_test)

#print("Accuracy(R2 score):",r2_score*100,'%')
#print("R score:",r)
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("RMSE", math.sqrt(mse))

reg1.fit(x, y)
reg2.fit(x, y)
reg3.fit(x, y)

model.fit(x,y)
y_predicted = model.predict(test_data)
print(y_predicted.shape)
# write to csv file
write_to_csv('predictedFromVotingRegressor.csv', y_predicted)




