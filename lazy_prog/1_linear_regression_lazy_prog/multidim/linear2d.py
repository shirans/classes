import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib


def parse_csv(csv):
    x_matrix = []
    y_matrix = []
    for line in open(csv):
        x1, x2, y = line.split(',')
        x_matrix.append([float(x1), float(x2), 1.0])  # add the bias term
        y_matrix.append(float(y))

    # let's turn X and Y into numpy arrays since that will be useful later
    x_matrix = np.array(x_matrix)
    y_matrix = np.array(y_matrix)
    return x_matrix, y_matrix


def pandas_csv(csv):
    data = pd.read_csv(csv, header=None, float_precision='round_trip')
    data = data.assign(a=pd.Series(np.ones(100)).values)
    x_matrix = data[[0, 1, 'a']].as_matrix()
    y_matrix = data[[2]].as_matrix()
    return x_matrix, y_matrix


def solve(x_mat, y_mat):
    w = np.linalg.solve(np.dot(x_mat.transpose(), x_mat), np.dot(x_mat.transpose(), y_mat))
    y_hat = np.dot(x_mat, w)
    return w, y_hat


def eval(w, y_mat, y_hat):
    d1 = y_mat - y_hat
    d2 = y_mat - y_mat.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    ##### plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)
    plt.show()
    print(r2)


csv = '/Users/shiran/workspace/classes/lazy_prog/1_linear_regression_lazy_prog/multidim/data_2d.csv'

X, Y = parse_csv(csv)
X2, Y2 = pandas_csv(csv)

w, yHat = solve(X, Y)
w2, yHat2 = solve(X2, Y2)
eval(w, y_mat=Y, y_hat=yHat)
print("")