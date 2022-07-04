import math
from sklearn.svm import SVR, LinearSVR
from sklearn.datasets import make_regression
import csv
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import time
from datetime import datetime
from sklearn.model_selection import train_test_split

def rmse(prediction, value):
    sum = 0
    for i in range(len(prediction)):
        sum = sum + (prediction[i]-value[i])**2
    return math.sqrt(sum/len(prediction))


my_data = pd.read_csv('HomeC.csv', parse_dates=True)
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
# print(home_data.info())

#defining input and output variables as x and y respectively
x = home_data.filter(items=['temperature','humidity', 'visibility', 'apparentTemperature', 'pressure',
                            'windSpeed', 'windBearing', 'dewPoint'])

y = home_data.filter(items=['House overall [kW]'])

x.dropna(inplace=True)
y.dropna(inplace=True)
# print(x.shape)
# print(y.shape)
# print(type(y), type(x))

#splitting the data into test and train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# x_train1, y_train1 = x_train[0:1000], y_train[0:1000]
# x_test1, y_test1 = x_test[0:10], y_test[0:10]


# print(type(x_train))

# svr_lin = SVR(kernel='linear', C=100, gamma='auto')
# svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
# svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
#                coef0=1)

# svr_lin.fit(x_train1, y_train1)
# svr_rbf.fit(x_train, y_train)
# svr_poly.fit(x_train, y_train)

# print("predict:", svr_lin.predict(x_test1))
# print("predict:", svr_rbf.predict(x_test))
# print("predict:", svr_poly.predict(x_test))
# print("value:", y_test1)

regr = LinearSVR(random_state=42, tol=1e-5)
regr.fit(x_train, y_train)
# LinearSVR(random_state=0, tol=1e-05)
# print(regr.coef_)
# print(regr.intercept_)
# print("predict:", regr.predict(x_test1))
# print("value:", y_test1)

prediction = regr.predict(x_test)
values = y_test.values.tolist()
print(rmse(prediction, values))

# clf = SVR(C=1.0, epsilon=0.2)
# clf.fit(x_train, y_train)
#
# prediction = clf.predict(x_test)
# print(rmse(prediction, values))



# x=[]
# y=[]
#
# testx=[]
# testy=[]
#
# threshold=500000
# data_size = 503910
#
# with open('HomeC.csv') as csvDataFile:
#     data = csv.reader(csvDataFile)
#     for n, row in enumerate(data):
#         if 0 < n < threshold:
#             x.append([float(row[19]),float(row[21]),float(row[22]),float(row[24]),
#                       float(row[25]),float(row[26]),float(row[28]),float(row[30])])
#             y.append(float(row[1]))
#         if threshold<=n<=data_size:
#             testx.append([float(row[19]), float(row[21]), float(row[22]), float(row[24]),
#                       float(row[25]),float(row[26]), float(row[28]), float(row[30])])
#             testy.append(float(row[1]))

# print(y)
# print(x)
# print(len(y), len(x))
# exit(0)
# regr = LinearSVR(random_state=0, tol=1e-5)
# regr.fit(x, y)
# LinearSVR(random_state=0, tol=1e-05)
# print(regr.coef_)
# print(regr.intercept_)
# print("predict:", regr.predict(testx))
# print("value:", testy)



