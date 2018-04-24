import matplotlib.pyplot as plt
import numpy as np

from lazy_prog.common.common import sigmoid, cross_entropy_mean, get_e_commerce_example_data, get_e_commerce_binary_data, \
    logistic_regression_with_test, classification_rate

import numpy as np
import matplotlib.pyplot as plt

# get the data
Xtrain, Ytrain, Xtest, Ytest = get_e_commerce_binary_data()


# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))


w, train_costs, test_costs = logistic_regression_with_test(Xtrain, Ytrain, Xtest, Ytest)
pYtrain = sigmoid(Xtrain.dot(w))
pYtest = sigmoid(Xtest.dot(w))

print("Final train classification_rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, np.round(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
