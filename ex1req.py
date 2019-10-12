import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt', sep=',', header=None)
data.columns = ["xs", "ys"]
plt.rcParams['figure.figsize'] = (12.0, 9.0)

X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values
plt.scatter(X, Y)
plt.show()

m = 0
c = 0
L = 0.01
iter = 1500
n = float(len(X))

h = c + m * X


def costfunc():
    jbm = 0
    for i in range(len(X)):
        jbm = sum( (h - Y ) ** 2)
    j = (1 / (2 * len(X))) * (jbm)
    return j

a = costfunc()
print(a)

for i in range(iter):
    h = m * X + c
    t0 = (1 / n) * sum(X * (h - Y))
    t1 = (1 / n) * sum(h - Y)
    m = m - L * t0
    c = c - L * t1
print(m, c)

h = m*X + c

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(h), max(h)], color='red') # predicted
plt.show()
