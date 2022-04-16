# Linear Regression Code Here
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from PreProcessing import preprocessing
from helper_functions import write_to_csv

train_data, test_data = preprocessing()
y = train_data['Rented Bike Count'].values
x = train_data.drop(['Rented Bike Count'], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
linearRegressor = LinearRegression()
linearRegressor.fit(x_train, y_train)
test_data = np.array(test_data.drop(['ID'], axis=1))
y_predicted = linearRegressor.predict(test_data)
print('hello')
# mse = mean_squared_error(y_test, y_predicted)
# r = r2_score(y_test, y_predicted)
# mae = mean_absolute_error(y_test, y_predicted)
# r2_score = linearRegressor.score(x_test, y_test)

# print("Accuracy(R2 score):", r2_score * 100, '%')
# print("R score:", r)
# print("Mean Squared Error:", mse)
# print("Mean Absolute Error:", mae)

# write to csv file
write_to_csv('predictedFromLRR.csv', y_predicted)
# with open('predictedFromLR.csv', 'w') as f:
#     f.write("ID,Rented Bike Count\n")
#     for i in range(len(y_predicted)):
#         f.write(str(i) + ',' + str(float(y_predicted[i])) + '\n')
