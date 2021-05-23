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
    
    def predict(self, X, threshold = 0.5):
        probability = self.hypothesis_function(self.w, X)
        print(probability)
        return (probability<threshold).astype(int)
        
        
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
            # print("h-y ", temp1.shape)
            dtheta = (alpha/m)*np.matmul(X.T,h-y)
            # print("temp ", temp.shape)
            theta = np.subtract(theta , dtheta)
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
y_pred = classifier.predict(X_test)

#%%

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
#%%

## Visualising the Training set results

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()