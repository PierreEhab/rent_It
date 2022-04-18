# Write predictions in csv file.
addedfrom sklearn.preprocessing import MinMaxScaler


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
