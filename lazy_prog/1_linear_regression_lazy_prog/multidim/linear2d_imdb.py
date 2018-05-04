import pandas as pd
import re
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

X = []
Y = []
csv = '/Users/shiran/workspace/datasets/tmdb_5000_movies.csv'
for line in open(csv):
    x1, x2, y = line.split(',')
    X.append([1.0, float(x1), float(x2)])  # add the bias term
    Y.append(float(y))


# let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)

w = np.linalg.solve(np.dot(X.transpose(), X), np.dot(X.transpose(), Y))

yHat = np.dot(X, w)

d1 = Y - yHat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print(r2)
##### plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()



#
#
# # let's plot the data to see what it looks like
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# d = pd.read_csv(csv, header=None)
# arr = d.as_matrix()
# X_2 = d[[0, 1]].as_matrix()
# Y_2 = d[[2]].as_matrix()
#
# print(X.shape)
# print(X_2.shape)
