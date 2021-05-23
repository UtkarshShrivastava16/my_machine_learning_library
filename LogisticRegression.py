# -*- coding: utf-8 -*-
"""
Created on Sat May 22 15:35:18 2021

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%

class LogisticRegression:
    def __init__(self, X, y, alpha = 0.01, iterations = 10000):
        self.X = X
        self.y = y
        self.alpha = alpha
        self. iterations = iterations
        self.w = self.gradient_descent()
        
    def hypothesis_function(self,theta,X):
        # print("theta ", theta.shape)
        # print("X ", X.shape)
        z = np.matmul(X, theta)
        # print("z", z.shape)
        h = 1/(1+np.exp(z)) 
        return h
    
    def predict(self, theta, X):
        y_pred = self.hypothesis_function(theta, X)
        return (y_pred>0.5).astype(int)
        
        
    def gradient_descent(self):
        
        X = self.X
        y = self.y
        y = y.reshape(len(y),1)
        theta = np.random.randn(len(X[0]),1)
        alpha = self.alpha
        iterations = self.iterations
        m = len(y)
        # print("theta ", theta.shape)
        # print("X ", X.shape)
        for i in range(iterations):
           
            h = self.hypothesis_function(theta, X)
            # print("h", h.shape)
            # print("y", y.shape)
            temp1 = h-y
            # print("h-y ", temp1.shape)
            
            temp = np.matmul(X.T,temp1)
            # print("temp ", temp.shape)
            theta = np.subtract(theta , (alpha/m)*temp)
        # print(theta.shape)
        return theta
    
#%%
    
## Importing the dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
## Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train

classifier = LogisticRegression(X_train, y_train)
y_pred = classifier.predict(classifier.w, X_test)

#%%

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
