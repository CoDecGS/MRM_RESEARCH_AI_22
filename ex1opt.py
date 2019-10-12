import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data2.txt', sep=',', header=None)
data.columns = ["x1s", "x2s", "ys"]
plt.rcParams['figure.figsize'] = (12.0, 9.0)

X1 = data.iloc[:, 0].values
X2 = data.iloc[:, 1].values
Y = data.iloc[:, 2].values
plt.scatter(X1, Y)
plt.scatter(X2, Y)
plt.show()
plt.show()


fX1 = (X1)/[max(X1) - min(X1)]
fX2 = (X2)/[max(X2) - min(X2)]
fY = (Y)/[max(Y) - min(Y)]

m = 0
l = 0
c = 0

L = 0.01
it = 1500

h = c + m * fX1 + l * fX2
n = float(len(fX1))


def costfuncmul():
    jbm = 0
    for i in range(len(fX1)):
        jbm = sum((h - fY) ** 2)
    j = (1 / (2 * len(fX1))) * (jbm)
    return j

a=costfuncmul()
print(a)


for i in range(it):
    h = c + m * fX1 + l * fX2
    t0 = (1 / n) * sum(fX1 * (h - fY))
    t1 = (1 / n) * sum(fX2 * (h - fY))
    t2 = (1 / n) * sum(h - fY)
    m = m - L * t0
    l = l - L * t1
    c = c - L * t2

print(m, l, c)

h = c + m * fX1 + l * fX2
plt.scatter(fX1, fY)
plt.plot([min(fX1), max(fX1)], [min(h), max(h)], color='red') # predicted
plt.show()

plt.scatter(fX2, fY)
plt.plot([min(fX2), max(fX2)], [min(h), max(h)], color='red') # predicted
plt.show()




