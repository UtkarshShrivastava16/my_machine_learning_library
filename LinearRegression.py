# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:54:08 2021

@author: hp
"""
import numpy as np

class LinearRegression:
    def __init__(self, X_train, y_train, gd = False, alpha = 0.01, iterations = 1000):
        self.X_train = X_train
        self.y_train = y_train
        
        if gd:
            self.alpha = alpha
            self.iterations = iterations
            self.w = self.gradient_descent()
        else:
            self.w = self.train()
    
    def train(self):
        X = self.X_train
        y = self.y_train
        #Normal Equation
        w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)),np.transpose(X)), y)
        return w
    
    def predict(self, X):
        y = np.matmul(X,self.w)
        return y
    
    def cost_function(self, X, y, theta):
        # Number of observations
        m = len(y)
        
        # Hypothesis
        h = np.matmul(X,theta)
        
        #Cost function
        J = (1/2*m)*np.sum(np.square((h - y)))
        return J
        
    def gradient_descent(self):
        
        X = self.X_train 
        y = self.y_train
        y = y.reshape(len(y),1)
        alpha = self.alpha
        iterations = self.iterations
        theta = np.random.randn(len(X[0]),1)
        
        #Number of observations
        m = len(y)
        #Cost function record
        cost_function_history = np.zeros([iterations, 1])
        theta_history = []
        # print("X ", X.shape)
        # print("y ", y.shape)
        # print("theta " , theta.shape)
        for i in range(iterations):
            h = np.matmul(X,theta)
            # print("h " , h.shape)
            diff = h-y
            # print("diff" , diff.shape)
            theta = theta - (alpha/m)*(np.matmul(X.T,diff))
            cost_function_history[i][0] = self.cost_function(X,y, theta)
            theta_history.append(theta)
            # print("theta", theta.shape)
        return theta