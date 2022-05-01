import numpy as np
import pandas as pd
from sklearn import  linear_model, metrics, pipeline, preprocessing
from helper_functions import write_to_csv, normalize_data, standardize_data
from sklearn.model_selection import train_test_split

def rmsle(y, y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

# Date Reading
data = pd.read_csv('data set/train.csv')
train_data = data.iloc[:-1000, :]
test_data = data.iloc[-1000:, :]
print(data.shape, train_data.shape, test_data.shape)

train_labels = train_data['Rented Bike Count'].values
train_data = train_data.drop(['Date', 'Rented Bike Count','Temperature(°C)'], axis = 1)
test_labels = test_data['Rented Bike Count'].values
test_data = test_data.drop(['Date', 'Rented Bike Count','Temperature(°C)'], axis = 1)

binary_data_columns = ['Holiday', 'Functioningday']
binary_data_indices = np.array([(column in binary_data_columns) for column in train_data.columns], dtype = bool)

categorical_data_columns = ['Seasons']
categorical_data_indices = np.array([(column in categorical_data_columns) for column in train_data.columns], dtype = bool)

numeric_data_columns = ['Dew point temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Hour',
                        'Visibility (10m)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']
numeric_data_indices = np.array([(column in numeric_data_columns) for column in train_data.columns], dtype = bool)

transformer_list = [
    # binary
    ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_data_indices])),

    # numeric
    ('numeric_variables_processing', pipeline.Pipeline(steps=[
        ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_data_indices])),
        ('scaling', preprocessing.StandardScaler(with_mean=0))
    ])),

    # categorical
    ('categorical_variables_processing', pipeline.Pipeline(steps=[
        ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),
        ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown='ignore'))
    ])),
]
'''y = data['Rented Bike Count'].values.reshape(-1, 1)
x = data.drop(['Rented Bike Count','Date','Temperature(°C)'], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=24)'''

regressor = linear_model.Lasso(max_iter = 2000)
estimator = pipeline.Pipeline(steps = [
    ('feature_processing', pipeline.FeatureUnion(transformer_list=transformer_list)),
    ('model_fitting', regressor)
    ]
)
print(type(train_data),type(train_labels))
estimator.fit(train_data, train_labels)
predicted = estimator.predict(test_data)

print("RMSLE: ", rmsle(test_labels, predicted))
print("MAE: ",  metrics.mean_absolute_error(test_labels, predicted))