# voting regressor Code Here
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

from sklearn.preprocessing import MinMaxScaler


def write_to_csv(file_name, predictions):
    with open(file_name, 'w') as f:
        f.write("ID,Rented Bike Count\n")
        for i in range(len(predictions)):
            f.write(str(i) + ',' + str(float(predictions[i])) + '\n')


def normalize_data(data):
    scaler = MinMaxScaler()
    scaler_fit = scaler.fit(data)
    data = scaler_fit.transform(data)
    return data

def preprocessing():
    # Date Reading
    train_data = pd.read_csv('data set/train.csv')
    test_data = pd.read_csv('data set/test.csv')

    # 1-Filter input features
    # print(data['Holiday'].unique()) #['No Holiday' 'Holiday']
    train_data['Holiday'] = train_data['Holiday'].replace(['Holiday'], 1)
    train_data['Holiday'] = train_data['Holiday'].replace(['No Holiday'], 0)
    test_data['Holiday'] = test_data['Holiday'].replace(['Holiday'], 1)
    test_data['Holiday'] = test_data['Holiday'].replace(['No Holiday'], 0)

    # print(data['Functioning Day'].unique()) #['Yes' 'No']
    train_data['Functioning Day'] = train_data['Functioning Day'].replace(['Yes'], 1)
    train_data['Functioning Day'] = train_data['Functioning Day'].replace(['No'], 0)
    test_data['Functioning Day'] = test_data['Functioning Day'].replace(['Yes'], 1)
    test_data['Functioning Day'] = test_data['Functioning Day'].replace(['No'], 0)
    train_data['Seasons'] = train_data['Seasons'].replace(['Winter'], 0)
    train_data['Seasons'] = train_data['Seasons'].replace(['Spring'], 1)
    train_data['Seasons'] = train_data['Seasons'].replace(['Summer'], 2)
    test_data['Seasons'] = test_data['Seasons'].replace(['Summer'], 2)
    test_data['Seasons'] = test_data['Seasons'].replace(['Autumn'], 3)
    train_data = train_data.drop(['Date'], axis=1)
    test_data = test_data.drop(['Date'], axis=1)
    # test_data = test_data.drop(['Dew point temperature(°C)'], axis=1)
    # train_data = train_data.drop(['Dew point temperature(°C)'], axis=1)
    # test_data = test_data.drop(['Seasons'], axis=1)
    # train_data = train_data.drop(['Seasons'], axis=1)
    test_data = test_data.drop(['Temperature(°C)'], axis=1)
    train_data = train_data.drop(['Temperature(°C)'], axis=1)
    corr_mat = train_data.corr()
    sn.heatmap(corr_mat, annot=True)
    plt.show()
    return train_data, test_data





train_data, test_data = preprocessing()

test_data = test_data.drop(['ID'], axis=1)
# test_data = test_data.drop(['Dew point temperature(c)'], axis=1)
# test_data = test_data.drop(['ID'], axis=1)
y = train_data['Rented Bike Count'].values
x = train_data.drop(['Rented Bike Count'], axis=1).values
x = normalize_data(x)
test_data = np.array(test_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=24)



#model = GradientBoostingRegressor()
#model.fit(x_train,y_train)
#test_data = np.array(test_data)
#y_predicted = model.predict(test_data)





pred = 0
reg1 = GradientBoostingRegressor()
reg2 = RandomForestRegressor()
reg3 = LinearRegression()

reg1.fit(x, y)
reg2.fit(x, y)
reg3.fit(x, y)

model= VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])



model.fit(x_train,y_train)
y_predicted = model.predict(test_data)
print(y_predicted.shape)
# write to csv file
write_to_csv('predictedFromVotingRegressor.csv', y_predicted)



y_pred= model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
#r2_score = VotingRegressor.score(x_test,y_test)

#print("Accuracy(R2 score):",r2_score*100,'%')
print("R score:",r)
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)

