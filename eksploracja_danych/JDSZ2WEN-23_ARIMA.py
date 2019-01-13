#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:24:49 2019

@author: nanokoper
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

df = pd.read_csv('/home/nanokoper/Pulpit/Projekt.csv')
df.columns = ['Date', 'Casheng', 'Cashother']
df['Date'] = df.Date.astype(str)
df['Casheng'] = df.Casheng.astype(float)
df['Cashother'] = df.Cashother.astype(float)
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df = df.set_index(['Date'])
print(df)

rolmean= df['Casheng'].rolling(window=12).mean()
rolstd = df['Casheng'].rolling(12).std()

print(rolmean, rolstd)

"""orig = plt.plot(df['Casheng'], color = 'b', label = 'Original')
mean = plt.plot(rolmean, color = 'r', label = 'Rolling Mean')
std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
plt.legend(loc='best')
plt.show()"""

df_logScale = np.log(df['Casheng'])
movAvg = df_logScale.rolling(window=12).mean()
movSTD = df_logScale.rolling(window=12).std()
"""plt.plot(df_logScale)
plt.plot(movAvg, color = 'r')"""

dffin = df_logScale - movAvg
dffin.dropna(inplace = True)



expdecweightavg = df_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
dfl = df_logScale - expdecweightavg

rolmeandfl= dfl.rolling(window=12).mean()
rolstddfl = dfl.rolling(12).std()

orig = plt.plot(dfl, color = 'b', label = 'Original')
mean = plt.plot(rolmeandfl, color = 'r', label = 'Rolling Mean')
std = plt.plot(rolstddfl, color = 'black', label = 'Rolling Std')
plt.legend(loc='best')
plt.show()

dfshift = df_logScale - df_logScale.shift()
dfshift.dropna(inplace=True)
decomposition = seasonal_decompose(df_logScale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

decomposedLogData = residual

lag_acf = acf(dfshift, nlags = 20)
#plt.plot(lag_acf)
lag_pacf = pacf(dfshift, nlags = 20, method = 'ols')
#plt.plot(lag_pacf)
print(dfshift)


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df_logScale, order=(0,1,13))
results_ARIMA = model.fit(disp=-1)

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
cumsum = predictions_ARIMA_diff.cumsum()

predARIMAlog = pd.Series(df_logScale.ix[0], index=df_logScale.index)
predARIMAlog = predARIMAlog.add(cumsum, fill_value = 0)

predARIMA = np.exp(predARIMAlog)
print(predARIMA)
x = results_ARIMA.forecast(steps = 24)
print("To jest przewidywanie")
print(x)












