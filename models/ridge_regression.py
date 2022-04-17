from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from helper_functions import write_to_csv
from PreProcessing import preprocessing


train_data, test_data = preprocessing()
y = train_data['Rented Bike Count'].values.reshape(-1, 1)
x = train_data.drop(['Rented Bike Count'], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

reg = Lasso(alpha=1.0)
reg.fit(x_train, y_train)
y_pred = reg.predict(test_data.drop(['ID'], axis=1))
write_to_csv('predictedFromRidgeReg.csv', y_pred)
#y_pred = reg.predict(x_test)
#print("MAE",mean_absolute_error(y_test,y_pred))