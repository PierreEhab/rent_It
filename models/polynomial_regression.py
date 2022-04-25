from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from PreProcessing import preprocessing
from helper_functions import standardize_data, normalize_data, write_to_csv
from sklearn.preprocessing import PolynomialFeatures

train_data, test_data = preprocessing()
test_data = test_data.drop(['ID'], axis=1)
# Get X and Y

x = train_data.drop(['Rented Bike Count'], axis=1).values
y = train_data['Rented Bike Count'].values.reshape(-1, 1)
# # Standardization
# x = standardize_data(x)

# x = normalize_data(x)
# test_data = normalize_data(test_data)

# Fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

print(len(x))
print(len(y))
# Visualizing the Polymonial Regression results
# def viz_polymonial():
#     plt.scatter(x, y, color='red')
#     plt.plot(x, pol_reg.predict(poly_reg.fit_transform(x)), color='blue')
#     plt.title('Truth or Bluff (Linear Regression)')
#     plt.xlabel('Position level')
#     plt.ylabel('Salary')
#     plt.show()
#     return
#
#
# viz_polymonial()
# Predicting a new result with Polymonial Regression
predictions = pol_reg.predict(poly_reg.fit_transform(test_data))
write_to_csv('predictedFromPolyReg.csv', predictions)
