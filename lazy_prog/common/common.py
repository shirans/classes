import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# so scripts from other folders can import this file
dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# Set this so that when np encounters error, it'll throw them and not just log them
np.seterr(all='raise')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# calculate the cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


# cross entropy with numpy sum
def cross_entropy_numpysum(T, pY):
    try:
        return -np.nansum(T * np.log(pY) + (1 - T) * np.log(1 - pY))
    except FloatingPointError:
        print "got runtime exception"


# Using the mean of the cross entropy of all the samples is helpful for gradient descent because
# it makes the learning rate insensitive to the number of samples.
# It's equivalent to multiplying by a constant.
# if you need to actual value of the log-likelyhood,
# do not use mean cause by definition it's the sum of the cross entropy
def cross_entropy_mean_nan(T, pY):
    return -np.nanmean(T * np.log(pY) + (1 - T) * np.log(1 - pY))


# cross entropy
def cross_entropy_mean(T, pY):
    return -np.mean(T * np.log(pY) + (1 - T) * np.log(1 - pY))


def print_data(Xb, t, w):
    print("Xb shape:", Xb.shape)
    print("targets shape:", t.shape)
    print("w shape:", w.shape)


def forward(X, W, b):
    return sigmoid(X.dot(W) + b)


def create_data_2_gaussian_clouds(N, D):
    n_per_class = N // 2
    # Create Xb - random N samples with D dimensions + bias -> Xb(N,D+1)
    X = np.random.randn(N, D)

    # center the first 50 points at (-2,-2)
    X[:n_per_class, :] = X[:n_per_class, :] - 2 * np.ones((n_per_class, D))

    # center the last 50 points at (2, 2)
    X[n_per_class:, :] = X[n_per_class:, :] + 2 * np.ones((n_per_class, D))

    # labels: first N_per_class are 0, last N_per_class are 1
    targets = np.array([0] * n_per_class + [1] * n_per_class)
    ones = np.ones((N, 1))

    Xb = np.concatenate((ones, X), axis=1)
    return Xb, targets


def print_cross_entropy_error_for_w(Xb, w, T):
    Y = sigmoid(Xb.dot(w))
    print(cross_entropy(T, Y))


def visualize_two_classes_separation(Xb, T, x_axis, y_axis, _label):
    plt.scatter(Xb[:, 1], Xb[:, 2], c=T, s=100, alpha=0.5)
    x_axis = np.linspace(-6, 6, 100)
    y_axis = -x_axis
    plt.plot(x_axis, y_axis)
    plt.show()


def get_e_commerce_example_data():
    df = pd.read_csv(dir_path + '/ecommerce_data.csv')

    # just in case you're curious what's in it
    # df.head()

    # easier to work with numpy array
    data = df.as_matrix()

    # shuffle it
    np.random.shuffle(data)

    # split features and labels
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)

    # one-hot encode the categorical data
    # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]  # non-categorical

    # one-hot
    for n in range(N):
        t = int(X[n, D - 1])
        X2[n, t + D - 1] = 1

    # method 2
    # Z = np.zeros((N, 4))
    # Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # # assign: X2[:,-4:] = Z
    # assert(np.abs(X2[:,-4:] - Z).sum() < 1e-10)

    # assign X2 back to X, since we don't need original anymore
    X = X2

    # split train and test
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Xtest = X[-100:]
    Ytest = Y[-100:]

    # normalize columns 1 and 2
    for i in (1, 2):
        m = Xtrain[:, i].mean()
        s = Xtrain[:, i].std()
        Xtrain[:, i] = (Xtrain[:, i] - m) / s
        Xtest[:, i] = (Xtest[:, i] - m) / s

    return Xtrain, Ytrain, Xtest, Ytest


def get_e_commerce_binary_data():
    # return only the data from the first 2 classes
    Xtrain, Ytrain, Xtest, Ytest = get_e_commerce_example_data()
    X2train = Xtrain[Ytrain <= 1]
    Y2train = Ytrain[Ytrain <= 1]
    X2test = Xtest[Ytest <= 1]
    Y2test = Ytest[Ytest <= 1]
    return X2train, Y2train, X2test, Y2test


def logistic_regression_with_test(xtrain, ytrain, xtest, ytest):
    [N, D] = xtrain.shape
    # randomly initialize weights
    w = np.random.randn(D)
    b = 0  # bias term

    train_costs = []
    test_costs = []
    learning_rate = 0.001
    # train loop
    for i in range(10000):
        pYtrain = forward(xtrain, w, b)
        pYtest = forward(xtest, w, b)

        ctrain = cross_entropy_mean(ytrain, pYtrain)
        ctest = cross_entropy_mean(ytest, pYtest)
        train_costs.append(ctrain)
        test_costs.append(ctest)

        # gradient descent
        w -= learning_rate * xtrain.T.dot(pYtrain - ytrain)
        b -= learning_rate * (pYtrain - ytrain).sum()
        if i % 1000 == 0:
            print(i, ctrain, ctest)

    return w, train_costs, test_costs


# calculate the accuracy
def classification_rate(Y, T):
    return np.mean(Y == T)
