# -*- coding: utf-8 -*-
"""
Created on Sun May 23 10:55:06 2021

@author: hp
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict 
#%%
class KNN:
    def __init__(self, X, y, k = 5):
        self.X = X
        self.y = y
        self.k = k
    
    def eucledian_distance(self, point1, point2):
        distance = 0
        
        for i in range(len(point1)):
            distance+=(point1[i]-point2[i])**2
            
        return math.sqrt(distance)
    
    def k_nearest_neighbors(self, point):
        X = self.X
        y = self.y
        nn = []
        
        # Store all neighbors to point
        # Store as -> [X value, y value, distance from point]
        for i in range(len(X)):
            nn.append([X[i], y[i], self.eucledian_distance(X[i], point)]) 
        
        #Sort on the basis of distance
        nn = sorted(nn, key=lambda item: item[2])
        
        # Return k neighbors
        return nn[:self.k]
    
    def classify(self, knn):
        y = [value[1] for value in knn]
        
        # Return max occuring value of y
        return max(set(y), key = y.count)
            
    
    def predict(self ,X_test):  
        result = []
        for i in X_test:
            knn = self.k_nearest_neighbors(i)
            result.append(self.classify(knn))
        return np.array(result)
        
        
        
    
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

classifier = KNN(X_train, y_train,  k = 5)
y_pred = classifier.predict(X_test)
#%%

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

#%%

## Visualising the Training set results

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

## Visualising the Test set results

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()