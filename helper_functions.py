# Write predictions in csv file.
def write_to_csv(file_name, predictions):
    with open(file_name, 'w') as f:
        f.write("ID,Rented Bike Count\n")
        for i in range(len(predictions)):
            f.write(str(i) + ',' + str(float(predictions[i])) + '\n')
