import matplotlib.pyplot as plt
import numpy as np

from lazy_prog.common.common import sigmoid, cross_entropy_mean, get_e_commerce_example_data, \
    get_e_commerce_binary_data, \
    logistic_regression_with_test, classification_rate

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 2], [3 ,4]])
ones = np.ones((2, 1))
Xb = np.concatenate((ones, X), axis=1)

w = np.array([1, 2, 3])
res = Xb.dot(w)
bias = res[:,-1]

w2 = np.array([1, 2, 3])
b = 1
res_2 = X.dot(w2)
bias2 = b*res_2
# get the data
Xtrain, Ytrain, Xtest, Ytest = get_e_commerce_binary_data()
Ytest = np.round(np.random.random(Ytest.shape[0]))


# Ytest = 1-Ytest

# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))


w, train_costs, test_costs = logistic_regression_with_test(Xtrain, Ytrain, Xtest, Ytest, 10000, 0.0001)
pYtrain = sigmoid(Xtrain.dot(w))
pYtest = sigmoid(Xtest.dot(w))

print("Final train classification_rate:", classification_rate(pYtrain, Ytrain))
print("Final test classification_rate:", classification_rate(pYtest, Ytest))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
