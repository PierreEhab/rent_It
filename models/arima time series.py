import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.dates import DateFormatter
from pmdarima import acf, pacf, ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

rcParams['figure.figsize'] = 10, 6
import pylab
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('data set/train.csv')
# print(train_data.head(), train_data.describe(), train_data.dtypes)
train_data['Date'] = pd.to_datetime(train_data['Date'], infer_datetime_format=True)
train_data['Holiday'] = train_data['Holiday'].replace(['Holiday'], 1)
train_data['Holiday'] = train_data['Holiday'].replace(['No Holiday'], 0)
train_data['Functioning Day'] = train_data['Functioning Day'].replace(['Yes'], 1)
train_data['Functioning Day'] = train_data['Functioning Day'].replace(['No'], 0)
train_data['Seasons'] = train_data['Seasons'].replace(['Winter'], 0)
train_data['Seasons'] = train_data['Seasons'].replace(['Spring'], 1)
train_data['Seasons'] = train_data['Seasons'].replace(['Summer'], 2)
train_data.drop(train_data.columns.difference(['Date', 'Rented Bike Count']), 1, inplace=True)
print('Shape of data', train_data.shape)
print(train_data.head())

train_data = train_data.sort_values(['Date', 'Rented Bike Count'])

indexedDataset = train_data.set_index(['Date'])

#train_data = train_data.groupby('Date')['Rented Bike Count'].mean()
print("after group by",train_data.head())


# Define the date format
# Create figure and plot space
fig, ax = plt.subplots(figsize=(12, 12))
date_form = DateFormatter("%y")
ax.xaxis.set_major_formatter(date_form)
plt.xlabel("Date")
plt.ylabel("Rented Bike Count")
plt.plot(indexedDataset)
plt.show()

# train_data['Rented Bike Count'].plot(figsize=(6000, 13))
# pylab.show()

# determining the rolling mean
rolmean = indexedDataset.rolling(window=356).mean()
rolstd = indexedDataset.rolling(window=365).std()
print('rolling\n', rolmean, rolstd)

orig = plt.plot(indexedDataset, color='blue', label='original')
mean = plt.plot(rolmean, color='red', label='rolling mean')
std = plt.plot(rolstd, color='black', label='rolling std')
plt.legend(loc='best')
plt.show()

from statsmodels.tsa.stattools import adfuller


def adf_test(dataset):

        '''dftest = adfuller(dataset, autolag='AIC')
        print("1. ADF : ", dftest[0])
        print("2. P-Value : ", dftest[1])
        print("3. Num Of Lags : ", dftest[2])
        print("4. Num Of Observations Used For ADF Regression:", dftest[3])
        print("5. Critical Values :")
        for key, val in dftest[4].items():
            print("\t", key, ": ", val)'''
        dftest = adfuller(dataset['Rented Bike Count'], autolag='AIC')

        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value

        print(dfoutput)



adf_test(indexedDataset)

indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)
plt.show()

movingAvg = indexedDataset_logScale.rolling(window=356).mean()
movingStd = indexedDataset_logScale.rolling(window=356).std()
plt.plot(movingAvg, color='red')
plt.show()

datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAvg
datasetLogScaleMinusMovingAverage.head(12)

# Remove NAN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)


def test_stationarity(timeseries):
    # Determine rolling statistics
    movingAverage = timeseries.rolling(window=365).mean()
    movingSTD = timeseries.rolling(window=365).std()

    # Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #timeseries.dropna(inplace=True)
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(timeseries['Rented Bike Count'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(datasetLogScaleMinusMovingAverage)

exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=365, min_periods=0, adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')
plt.show()

datasetLogScaleMinusExponentialMovingAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusExponentialMovingAverage)

datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)
plt.show()
datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)
print(datasetLogDiffShifting.head(),datasetLogDiffShifting.tail())

decomposition = seasonal_decompose(indexedDataset_logScale,model='additive')

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(411)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(411)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
# there can be cases where an observation simply consisted of trend & seasonality. In that case, there won't be
# any residual component & that would be a null or NaN. Hence, we also remove such cases.
decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)

# ACF & PACF plots

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

# Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

# Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')

plt.tight_layout()
plt.show()

# AR Model
# making order=(2,1,0) gives RSS=1.5023
model = ARIMA(indexedDataset_logScale, order=(2, 1, 0))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_AR.fittedvalues - datasetLogDiffShifting['#Passengers']) ** 2))
plt.show()
print('Plotting AR model')

# MA Model
model = ARIMA(indexedDataset_logScale, order=(0, 1, 2))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_MA.fittedvalues - datasetLogDiffShifting['#Passengers']) ** 2))
pkt.show()
print('Plotting MA model')

# AR+I+MA = ARIMA model
model = ARIMA(indexedDataset_logScale, order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['#Passengers']) ** 2))
plt.show()
print('Plotting ARIMA model')

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())
# Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum)

predictions_ARIMA_log = pd.Series(indexedDataset_logScale['#Passengers'].iloc[0], index=indexedDataset_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
print(predictions_ARIMA_log.head())

# Inverse of log is exp.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)
plt.show()

print(indexedDataset_logScale)

# We have 144(existing data of 12 yrs in months) data points.
# And we want to forecast for additional 120 data points or 10 yrs.
results_ARIMA.plot_predict(1, 264)
# x=results_ARIMA.forecast(steps=120)
