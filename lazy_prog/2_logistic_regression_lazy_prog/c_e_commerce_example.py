import numpy as np
import matplotlib.pyplot as plt
from common import sigmoid, cross_entropy, cross_entropy_mean, cross_entropy_numpysum, get_e_commerce_example_data


def get_binary_data():
    # return only the data from the first 2 classes
    Xtrain, Ytrain, Xtest, Ytest = get_e_commerce_example_data()
    X2train = Xtrain[Ytrain <= 1]
    Y2train = Ytrain[Ytrain <= 1]
    X2test = Xtest[Ytest <= 1]
    Y2test = Ytest[Ytest <= 1]
    return X2train, Y2train, X2test, Y2test


# get the data
Xtrain, Ytrain, Xtest, Ytest = get_binary_data()

# randomly initialize weights
D = Xtrain.shape[1]
W = np.random.randn(D)
b = 0  # bias term


# make predictions

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)


# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)


# train loop
train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy_mean(Ytrain, pYtrain)
    ctest = cross_entropy_mean(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent
    # In traditional machine learning it's typical to absorb the bias term.
    # In deep learning it's typical to keep it separate.
    # Both ways are valid and correct.
    W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate * (pYtrain - Ytrain).sum()
    if i % 1000 == 0:
        print(i, ctrain)
        print(i, ctrain, ctest)

print("Final train classification_rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, np.round(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()

# An example for making a prediction:
X1 = np.array([0, 1, 1, 0, 1, 6, 7, 8])
pred = sigmoid(X1.dot(W))
print("prediction:", '{0:f}'.format(pred), "classification:", np.round(pred))
