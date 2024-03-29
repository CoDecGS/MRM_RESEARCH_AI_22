# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:28:27 2019

@author: GHANASHYAM
"""

from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones, X)) #add the intercept
(m,n) = X.shape

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunctionReg(theta, X, y, lmbda):
    m = len(y)
    temp1 = np.multiply(y, np.log(sigmoid(np.dot(X, theta))))
    temp2 = np.multiply(1-y, np.log(1-sigmoid(np.dot(X, theta))))
    return np.sum(temp1 + temp2) / (-m) + np.sum(theta[1:]**2) * lmbda / (2*m)

def gradRegularization(theta, X, y, lmbda):
    m = len(y)
    temp = sigmoid(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / m + theta * lmbda / m
    temp[0] = temp[0] - theta[0] * lmbda / m
    return temp

lmbda = 0.1
k = 10
theta = np.zeros((k,n)) #inital parameters

for i in range(k):
    digit_class = i if i else 10
    theta[i] = opt.fmin_cg(f = costFunctionReg, x0 = theta[i],  fprime = gradRegularization, args = (X, (y == digit_class).flatten(), lmbda), maxiter = 50)
print(digit_class)    
pred = np.argmax(X @ theta.T, axis = 1)
pred = [e if e else 10 for e in pred]
print(np.mean(pred == y.flatten()) * 100)