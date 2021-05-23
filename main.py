# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:18:00 2021

@author: hp
"""
## Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%
# =============================================================================
# 1. Linear Regression
# =============================================================================
from LinearRegression import LinearRegression
## Importing the dataset

## Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# =================================Using Normal Equation - O(n^3)==============
# 
# regressor = LinearRegression(X_train,y_train)
# y_pred = regressor.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# =============================================================================

# =====================================Using gradient descent - O(n^2)=========
# 
# regressor = LinearRegression(X_train, y_train)
# y_pred = regressor.predict(X_test)
# 
# ## Visualising the Training set results
# 
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
# 
# ## Visualising the Test set results
# 
# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
# =============================================================================

#%%

