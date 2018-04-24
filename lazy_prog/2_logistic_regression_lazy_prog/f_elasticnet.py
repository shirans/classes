import matplotlib.pyplot as plt
import numpy as np

from lazy_prog.common.common import sigmoid

# ElasticNet is the name of adding both L1 and L2 normalization


N = 50
D = 50

# uniformly distributed numbers between -5, +5
X = (np.random.random((N, D)) - 0.5) * 10

# true weights - only the first 3 dimensions of X affect Y. We wish the model to learn only the first 3 mean anythin
true_w = np.array([1, 0.5, -0.5] + [0] * (D - 3))

# generate Y - add noise with variance 0.5
Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N) * 0.5))

# perform gradient descent to find w
costs = []  # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D)  # randomly initialize w
learning_rate = 0.001
# l1 = 3.0  # try different values - what effect does it have on w?
l1 = 10.0  # try different values - what effect does it have on w?
l2 = 0.01
for t in range(5000):
    # update w
    Yhat = sigmoid(X.dot(w))
    w = w - learning_rate * (X.T.dot(Yhat - Y) + l1 * np.sign(w))

    # find and store the cost
    cost = -(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat)).mean() + l1 * np.abs(w).mean() + l2*w
    costs.append(cost)

# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# plot our w vs true w
plt.plot(true_w, label='true w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()
