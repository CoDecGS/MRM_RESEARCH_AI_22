# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:42:25 2019

@author: GHANASHYAM
"""

from scipy.io import loadmat
import numpy as np

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

weights = loadmat('ex3weights.mat')
weights.keys()

print('X: {} (with intercept)'.format(X.shape))
print('y: {}'.format(y.shape))

theta1, theta2 = weights['Theta1'], weights['Theta2']

print('theta1: {}'.format(theta1.shape))
print('theta2: {}'.format(theta2.shape))

m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones, X)) 
(m,n) = X.shape

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(theta1, theta2, X):
    z2 = theta1.dot(X.T)
    a2 = np.c_[np.ones((X.shape[0],1)), sigmoid(z2).T]
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
        
    return(np.argmax(a3, axis=1)+1)

pred = predict(theta1, theta2, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))