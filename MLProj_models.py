# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:41:04 2020

"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 
import seaborn as sns
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


my_data = pd.read_csv('G:\My Drive\Spring2020\CS6316\Data\HomeC.csv', parse_dates=True)
# print(my_data.info())


# df[['h','m','s']] = pd.DataFrame([(x.hour, x.minute, x.second) for x in df['time']])





# excluding columns with 'object' type 
home_data = my_data.select_dtypes(exclude=['object'])


# converting unix epoch timestamp into datatime 
# print( ' start ' , time.strftime('%Y-%m-%d %H:%S', time.localtime(1451624400)))
# print( ' start ' , time.strftime('%Y-%m-%d %H:%S', time.localtime(1451624401)))

# defining a range time based on the start point and period
time_index = pd.date_range('2016-01-01 05:00', periods=503911,  freq='min')  
time_index = pd.DatetimeIndex(time_index)
home_data['H'] = [x.hour for x in time_index]
home_data['M'] = [x.minute for x in time_index]
home_data = home_data.set_index(time_index)
# print(home_data.info())

#defining input and output variables as x and y respectively
x = home_data.filter(items=['temperature','humidity', 'visibility', 'apparentTemperature', 'pressure',
                            'windSpeed', 'windBearing', 'dewPoint', 'H'])
# print(np.any(np.isnan(x)))

y = home_data.filter(items=['House overall [kW]'])
# print(x.shape)
# print(y.shape)
print(type(y), type(x))
# print(x.isnull())

x.dropna(inplace=True)
# print(x.isnull())
y.dropna(inplace=True)


#splitting the data into test and train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# ------------------------------------------------------------------------------------
# defining the model
reg = LinearRegression()

# cross validation
cv_results = cross_val_score(reg, x, y, cv=5)
# print(cv_results)


# fitting the model on train data set
reg.fit(x_train, y_train)

# prediction on test data set
y_pred = reg.predict(x_test)

# model accuracy
accuracy = reg.score(x_test, y_test)
print(accuracy)

# Root Mean Squared Error(RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
# ------------------------------------------------------------------------------------
# # ridge regression in scikit learn
# from sklearn.linear_model import Ridge

# # splitting the data into test and train set
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # defining the model
# ridge = Ridge(alpha=0.1, normalize=True)

# # cross validation
# # cv_results = cross_val_score(ridge, X, y, cv=5)
# # print(cv_results)

# # fitting the model on train data set
# ridge.fit(x_train, y_train)

# # prediction on test data set
# y_pred = ridge.predict(x_test)

# # model accuracy
# accuracy = ridge.score(x_test, y_test)
# print(accuracy)

# # # Root Mean Squared Error(RMSE)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(rmse)
# -------------------------------------------------------------------------------------
# # Lasso regression in sickit learn
# from sklearn.linear_model import Lasso

# # splitting the data into test and train set
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # defining the model
# lasso = Lasso(alpha=0.1, normalize=True)

# # cross validation
# cv_results = cross_val_score(lasso, x, y, cv=5)
# print(cv_results)

# # fitting the model on train data set
# lasso.fit(x_train, y_train)

# # prediction on test data set
# y_pred = lasso.predict(x_test)

# # model accuracy
# accuracy = lasso.score(x_test, y_test)
# print(accuracy)

# # # Root Mean Squared Error(RMSE)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(rmse)
# ----------------------------------------------------------------------------------
