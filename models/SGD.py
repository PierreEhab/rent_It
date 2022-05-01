import numpy as np
import pandas as pd
from sklearn import  linear_model, metrics, pipeline, preprocessing

# Date Reading
train_data = pd.read_csv('data set/train.csv')
test_data = pd.read_csv('data set/test.csv')

train_labels = train_data['Rented Bike Count'].values
train_data = train_data.drop(['Date', 'Rented Bike Count', 'Temperature(°C)'], axis = 1)
test_data = test_data.drop(['Date', 'Temperature(°C)'], axis = 1)

binary_data_columns = ['Holiday', 'Functioningday']
binary_data_indices = np.array([(column in binary_data_columns) for column in train_data.columns], dtype = bool)

categorical_data_columns = ['season', 'weather', 'month']
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

regressor = linear_model.Lasso(max_iter = 2000)
estimator = pipeline.Pipeline(steps = [
    ('feature_processing', pipeline.FeatureUnion(transformer_list=transformer_list)),
    ('model_fitting', regressor)
    ]
)

estimator.fit(train_data, train_labels)
predicted = estimator.predict(test_data)

print("RMSLE: ", rmsle(test_labels, predicted))
print("MAE: ",  metrics.mean_absolute_error(test_labels, predicted))