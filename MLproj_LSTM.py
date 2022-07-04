# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:42:27 2020

"""

from math import sqrt
import pandas as pd
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM


my_data = pd.read_csv('G:\My Drive\Spring2020\CS6316\Data\HomeC.csv', parse_dates=True)
# print(my_data.info())

# excluding columns with 'object' type 
home_data = my_data.select_dtypes(exclude=['object'])


# converting unix epoch timestamp into datatime 
# print( ' start ' , time.strftime('%Y-%m-%d %H:%S', time.localtime(1451624400)))
# print( ' start ' , time.strftime('%Y-%m-%d %H:%S', time.localtime(1451624401)))

# defining a range time based on the start point and period
time_index = pd.date_range('2016-01-01 05:00', periods=503911,  freq='min')  
time_index = pd.DatetimeIndex(time_index)
home_data = home_data.set_index(time_index)
values = home_data
values = values.filter(items=['House overall [kW]', 'temperature','humidity', 'visibility', 'apparentTemperature', 'pressure',
                             'windSpeed', 'windBearing', 'dewPoint'])
values.dropna(inplace=True)
# print(home_data.info())
print(values.columns.get_loc('House overall [kW]'))
print(values.columns.get_loc('temperature'))
print(values.columns.get_loc('humidity'))
print(values.columns.get_loc('visibility'))
print(values.columns.get_loc('apparentTemperature'))
print(values.columns.get_loc('pressure'))
print(values.columns.get_loc('windSpeed'))
print(values.columns.get_loc('windBearing'))
print(values.columns.get_loc('dewPoint'))

#defining input and output variables as x and y respectively
# values = home_data.filter(items=['temperature','humidity', 'visibility', 'apparentTemperature', 'pressure',
#                             'windSpeed', 'windBearing', 'dewPoint'])
# print(np.any(np.isnan(x)))

# y = home_data.filter(items=['House overall [kW]'])
# print(x.shape)
# print(y.shape)
# print(type(y), type(x))
# print(x.isnull())

# x.dropna(inplace=True)
# print(x.isnull())
# y.dropna(inplace=True)


# --------------------------------------------------------------------------------------------

# creating features and target arrays
# file = pd.read_csv('file.csv', header=0, index_col=0)
# values = file.values
# feature = file.drop(['time', 'temperature', 'month', 'day', 'minute'], axis=1)
# X = feature.values
# y = file['temperature'].values

# print(int(len(values)*0.7))
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
# dataset = read_csv('pollution.csv', header=0, index_col=0)
# values = dataset.values
# integer encode direction
# encoder = LabelEncoder()
# values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
# values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[10,11,12,13,14,15,16,17]], axis=1, inplace=True)
# print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = int(len(values)*0.7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics = ['accuracy'])
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# test_X1 = np.delete(test_X, 6, 1)
# inv_yhat = np.insert((test_X[:, 1:]))
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


