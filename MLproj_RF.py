# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:42:33 2020

@author: mahsa
"""

from pathlib import Path
import pandas as pd
import numpy as np
import csv
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
# creating features and target arrays
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
# home_data['H'] = [x.hour for x in time_index]
# home_data['M'] = [x.minute for x in time_index]
home_data = home_data.set_index(time_index)
# print(home_data.info())

#defining input and output variables as x and y respectively
X = home_data.filter(items=['temperature','humidity', 'visibility', 'apparentTemperature', 'pressure',
                            'windSpeed', 'windBearing', 'dewPoint'])
# print(np.any(np.isnan(x)))

y = home_data.filter(items=['House overall [kW]'])
# print(x.shape)
# print(y.shape)
print(type(y), type(X))
# print(x.isnull())

X.dropna(inplace=True)
# print(x.isnull())
y.dropna(inplace=True)
#
y = y.values.ravel()
feature_list = list(X.columns)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
errors = abs(predictions - y_test)
# print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
#  model accuracy
accuracyy = rf.score(X_test, y_test)
print(accuracyy)

# visualizing a singlr decision tree
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')
# --------------------------------------------------------------------------------------------
# important features
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# --------------------------------------------------------------------------------------------------
# train the model using just the important features
# New random forest with only the two most important variables
# rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# # Extract the two most important features
# important_indices = [feature_list.index('humidity'), feature_list.index('outside dewpt'), feature_list.index('outside windchill')]
# train_important = X_train[:, important_indices]
# test_important = X_test[:, important_indices]
# # Train the random forest
# rf_most_important.fit(train_important, y_train)
# # Make predictions and determine the error
# predictions = rf_most_important.predict(test_important)
# errors = abs(predictions - y_test)
# # Display the performance metrics
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# mape = np.mean(100 * (errors / y_test))
# accuracy = 100 - mape
# print('Accuracy:', round(accuracy, 2), '%.')
# accuracyy = rf_most_important.score(test_important, y_test)
# print(accuracyy)
# ------------------------------------------------------------------------------------------
# plotting the results
# Import matplotlib for plotting and use magic command for Jupyter Notebooks

# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation=90, size = 12)
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.show()
plt.savefig('feature_importance.png')
# plt.xticks(range(len(names)), names, rotation =10, size = 4)
# -------------------------------------------------------------------------------------------
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(X_train, y_train)
predictions = rf_small.predict(X_test)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')
