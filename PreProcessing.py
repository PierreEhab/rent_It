
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import average
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocessing():
    # Date Reading
    data = pd.read_csv('data set/train.csv')

    #1-Filter input features
    #print(data['Holiday'].unique()) #['No Holiday' 'Holiday']
    data['Holiday'] = data['Holiday'].replace(['Holiday'],1)
    data['Holiday'] = data['Holiday'].replace(['No Holiday'],0)

    # print(data['Functioning Day'].unique()) #['Yes' 'No']
    data['Functioning Day'] = data['Functioning Day'].replace(['Yes'],1)
    data['Functioning Day'] = data['Functioning Day'].replace(['No'],0)

    data=data.drop(['Date'],axis=1)
    #print(data.head())

    # 2- Get X and Y

    Y = data['Rented Bike Count'].values.reshape(-1, 1)

    # Apply LabelEncoding
    label_encoder=LabelEncoder()
    data['Seasons']=label_encoder.fit_transform(data['Seasons'])
    X = data.drop(['Rented Bike Count'], axis=1).values

    return X,Y
preprocessing()