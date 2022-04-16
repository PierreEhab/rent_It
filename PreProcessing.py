import pandas as pd


def preprocessing():
    # Date Reading
    train_data = pd.read_csv('data set/train.csv')
    test_data = pd.read_csv('data set/test.csv')
    # print(data)

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

    return train_data, test_data
