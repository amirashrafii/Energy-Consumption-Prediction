# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:03:48 2020

"""

import pandas as pd 
import matplotlib.pyplot as plt
# %matplotlib inline 
import seaborn as sns
import time
from datetime import datetime
from sklearn.model_selection import train_test_split

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
home_data['H'] = [x.hour for x in time_index]
home_data['M'] = [x.minute for x in time_index]
home_data = home_data.set_index(time_index)
# print(home_data.info())

#defining input and output variables as x and y respectively
x = home_data.filter(items=['temperature','humidity', 'visibility', 'apparentTemperature', 'pressure',
                            'windSpeed', 'windBearing', 'dewPoint', 'H', 'M'])
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
