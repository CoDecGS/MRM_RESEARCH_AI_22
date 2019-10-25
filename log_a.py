# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:41:05 2019

@author: GHANASHYAM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('ex2data1.txt', sep=',', header=None)
data.columns = ["Exam_1", "Exam_2", "sel"]
plt.rcParams['figure.figsize'] = (12.0, 9.0)
X = data.iloc[:, :-1]
X1 = data.iloc[:, 0]
X2 = data.iloc[:, 1]
y = data.iloc[:, -1]

admitted = data.loc[y == 1]
not_admitted = data.loc[y == 0]

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=20, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=20, label='Not Admitted')
plt.legend()
plt.show()

alpha = 0.01
num_iters = 30000

def sigmoid(x):
  return 1/(1+np.exp(-x))

def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta))) + np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))
    return J

def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))

(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
print(X)
y = y[:, np.newaxis]
theta = np.zeros((n+1,1))
print(theta)
J = costFunction(theta, X, y)
print(J)

#google searched optimization function and got this as in the programing assignment it's given to use matlab's inbuilt function
temp = opt.fmin_tnc(func = costFunction, x0 = theta.flatten(),fprime = gradient, args = (X, y.flatten()))

topt = temp[0]
print(topt)





