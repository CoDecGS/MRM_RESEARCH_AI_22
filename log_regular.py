# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 03:20:46 2019

@author: GHANASHYAM
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import scipy.optimize as opt

data = pd.read_csv('ex2data2.txt', sep=',', header=None)
data.columns = ["Exam_1", "Exam_2", "sel"]
plt.rcParams['figure.figsize'] = (12.0, 9.0)
X = data.iloc[:, :-1]
X1 = data.iloc[:, 0]
X2 = data.iloc[:, 1]
y = data.iloc[:, -1]

def sigmoid(x):
    return (1/(1+(np.exp(-x))))


def h(theta,X): 
    return sigmoid(np.dot(x,theta))
mlambda = 0.

x = np.ones([118,1])
print(x)
for i in range(7):
    for j in range(7-i):
        if i==0 and j==0:
            continue
        else:
            nw1 = np.array([X1**i]).reshape(118,1)
            nw2 = np.array([X2**j]).reshape(118,1)
            nw = nw1*nw2
            x = np.append(x,nw,axis=1)

def computeCost(theta,X,y,mlambda = 0.): 
  
  
    term1 = np.dot(-np.array(y).T,np.log(h(theta,X)))
    term2 = np.dot((1-np.array(y)).T,np.log(1-h(theta,X)))
    regterm = (mlambda/2) * np.sum(np.dot(theta[1:].T,theta[1:])) 
    return float( (1./118) * ( np.sum(term1 - term2) + regterm ) )

theta = np.zeros((x.shape[1],1))
computeCost(theta,x,y)

def gradient(theta, X, y, mlambda):
    m = len(y)
    g = np.zeros([m,1])
    g = (1/m) * x.T @ (sigmoid(x @ theta) - y)
    g[1:] = g[1:] + (mlambda / m) * theta[1:]
    return g

temp = opt.fmin_tnc(func = computeCost, x0 = theta, fprime = gradient, args = (X, y, mlambda))
theta = temp[0]


print(theta)
print(computeCost(theta,x,y))