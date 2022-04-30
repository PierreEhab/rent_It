import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def RemoveOutliers(data):

    toRemoveList=['Wind speed (m/s)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)']
    for i in toRemoveList:
        q75, q25 = np.percentile(data.loc[:, i], [75, 25])  # Divide data into 75%quantile and 25%quantile.
        iqr = q75 - q25  # Inter quantile range
        min = q25 - (iqr * 1.5)  # inner fence
        max = q75 + (iqr * 1.5)  # outer fence
        data.loc[data.loc[:, i] < min, :i] = np.nan  # Replace with NA
        data.loc[data.loc[:, i] > max, :i] = np.nan  # Replace with NA

    return data



def RemoveOutliersToMean(train_data, test_data):
    '''
    print(train_data.head)
   list =train_data.columns.values.tolist()
   # Set the figure size
   plt.rcParams["figure.figsize"] = [7.50, 3.50]
   plt.rcParams["figure.autolayout"] = True
   data =train_data
   ax = data[list].plot(kind='box', title='boxplot')
   plt.show()
   for i in list:
       ax = data[i].plot(kind='box', title='boxplot')
       plt.show()
   '''

    filtered_train_data = train_data.drop(['Holiday', 'Functioning Day', 'Rented Bike Count'], axis=1)
    filtered_train_data=RemoveOutliers(filtered_train_data)

    filtered_train_data['Wind speed (m/s)'] = filtered_train_data['Wind speed (m/s)'].fillna(filtered_train_data['Wind speed (m/s)'].mean())
    filtered_train_data['Solar Radiation (MJ/m2)'] = filtered_train_data['Solar Radiation (MJ/m2)'].fillna(
        filtered_train_data['Solar Radiation (MJ/m2)'].mean())
    filtered_train_data['Rainfall(mm)'] = filtered_train_data['Rainfall(mm)'].fillna(filtered_train_data['Rainfall(mm)'].mean())
    filtered_train_data['Snowfall (cm)'] = filtered_train_data['Snowfall (cm)'].fillna(filtered_train_data['Snowfall (cm)'].mean())

    train_data['Wind speed (m/s)']=filtered_train_data['Wind speed (m/s)']
    train_data['Solar Radiation (MJ/m2)']= filtered_train_data['Solar Radiation (MJ/m2)']
    train_data['Rainfall(mm)']= filtered_train_data['Rainfall(mm)']
    train_data['Snowfall (cm)']= filtered_train_data['Snowfall (cm)']

    filtered_test_data = test_data.drop(['Holiday', 'Functioning Day'], axis=1)
    filtered_test_data = RemoveOutliers(filtered_test_data)

    filtered_test_data['Wind speed (m/s)'] = filtered_test_data['Wind speed (m/s)'].fillna(filtered_test_data['Wind speed (m/s)'].mean())
    filtered_test_data['Solar Radiation (MJ/m2)'] = filtered_test_data['Solar Radiation (MJ/m2)'].fillna(
        filtered_test_data['Solar Radiation (MJ/m2)'].mean())
    filtered_test_data['Rainfall(mm)'] = filtered_test_data['Rainfall(mm)'].fillna(filtered_test_data['Rainfall(mm)'].mean())
    filtered_test_data['Snowfall (cm)'] = filtered_test_data['Snowfall (cm)'].fillna(filtered_test_data['Snowfall (cm)'].mean())

    test_data['Wind speed (m/s)'] = filtered_test_data['Wind speed (m/s)']
    test_data['Solar Radiation (MJ/m2)']= filtered_test_data['Solar Radiation (MJ/m2)']
    test_data['Rainfall(mm)']= filtered_test_data['Rainfall(mm)']
    test_data['Snowfall (cm)'] = filtered_test_data['Snowfall (cm)']

    return train_data, test_data


def RemoveOutliersToMedian(train_data, test_data ):

    filtered_train_data = train_data.drop(['Holiday', 'Functioning Day', 'Rented Bike Count'], axis=1)
    filtered_train_data=RemoveOutliers(filtered_train_data)

    filtered_train_data['Wind speed (m/s)'] = filtered_train_data['Wind speed (m/s)'].fillna(filtered_train_data['Wind speed (m/s)'].median())
    filtered_train_data['Solar Radiation (MJ/m2)'] = filtered_train_data['Solar Radiation (MJ/m2)'].fillna(
        filtered_train_data['Solar Radiation (MJ/m2)'].median())
    filtered_train_data['Rainfall(mm)'] = filtered_train_data['Rainfall(mm)'].fillna(filtered_train_data['Rainfall(mm)'].median())
    filtered_train_data['Snowfall (cm)'] = filtered_train_data['Snowfall (cm)'].fillna(filtered_train_data['Snowfall (cm)'].median())

    train_data['Wind speed (m/s)']=filtered_train_data['Wind speed (m/s)']
    train_data['Solar Radiation (MJ/m2)']= filtered_train_data['Solar Radiation (MJ/m2)']
    train_data['Rainfall(mm)']= filtered_train_data['Rainfall(mm)']
    train_data['Snowfall (cm)']= filtered_train_data['Snowfall (cm)']

    filtered_test_data = test_data.drop(['Holiday', 'Functioning Day'], axis=1)
    filtered_test_data = RemoveOutliers(filtered_test_data)

    filtered_test_data['Wind speed (m/s)'] = filtered_test_data['Wind speed (m/s)'].fillna(filtered_test_data['Wind speed (m/s)'].median())
    filtered_test_data['Solar Radiation (MJ/m2)'] = filtered_test_data['Solar Radiation (MJ/m2)'].fillna(
        filtered_test_data['Solar Radiation (MJ/m2)'].median())
    filtered_test_data['Rainfall(mm)'] = filtered_test_data['Rainfall(mm)'].fillna(filtered_test_data['Rainfall(mm)'].median())
    filtered_test_data['Snowfall (cm)'] = filtered_test_data['Snowfall (cm)'].fillna(filtered_test_data['Snowfall (cm)'].median())

    test_data['Wind speed (m/s)'] = filtered_test_data['Wind speed (m/s)']
    test_data['Solar Radiation (MJ/m2)']= filtered_test_data['Solar Radiation (MJ/m2)']
    test_data['Rainfall(mm)']= filtered_test_data['Rainfall(mm)']
    test_data['Snowfall (cm)'] = filtered_test_data['Snowfall (cm)']

    return train_data, test_data