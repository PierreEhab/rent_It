from PreProcessing import preprocessing

train_data, test_data = preprocessing()

# Get X and Y

x = train_data.drop(['Rented Bike Count'], axis=1).values

y = train_data['Rented Bike Count'].values.reshape(-1, 1)


print(x)
print(y)
