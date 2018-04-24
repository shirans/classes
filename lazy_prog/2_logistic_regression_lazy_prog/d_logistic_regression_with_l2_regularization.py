import numpy as np

from lazy_prog.common.common import sigmoid, cross_entropy, create_data_2_gaussian_clouds

# the goal - dealing with overfitting.
# One reason for over-fitting is irrelevant inputs.
# Another issue is that with out cost function, the ideal weights are infinite

# definition of odss: P(Event) / P (~Event). In our case: P(Y= 1 | X) / P(Y=0 | X)
# In practice we are doing linear regression on the log(odds)

# We add to the cost function a "prior believe" that the weights W should be small and centered around 0.
# Meaning: W~(0,1/g), meaning the variance of W is 1/g.
# Since the posterior ~= likelyhood + prior https://en.wikipedia.org/wiki/Posterior_probability
# The outcome is that in logistic regression we're maximizing the likelyhood, and with regularization we're maximizing
# the posterior, which is also called maximizing a posterior MAP estimation


# Another problem is: we usually want D to be much smaller than N.
# If it's not the case, we want to make the model choose only the meaninful features.
# so that most of W is zero.

# L1 regularization is also called Lasso regression, and L2 is also called ridge regression
# ==========================================================================================



N = 100
D = 2

Xb, T = create_data_2_gaussian_clouds(N, D)

# randomly initialize the weights
w = np.random.randn(D + 1)
w_l2 = w
w_normal = w
# calculate the model output
z = Xb.dot(w)
Y = sigmoid(z)
Y_l2 = Y
Y_normal = Y
# let's do gradient descent 100 times
learning_rate = 0.1
smoothing_parameter = 0.1

for i in range(100):
    if i % 10 == 0:
        print("entopy for i normal:", i, " : ", cross_entropy(T, Y_normal))
        print("entopy for i l2:", i, " : ", cross_entropy(T, Y_l2))

    w_l2 = w_l2 + learning_rate * (Xb.T.dot(T - Y_l2) - smoothing_parameter * w_l2)
    Y_l2 = sigmoid(Xb.dot(w_l2))

    w_normal = w_normal + learning_rate * (Xb.T.dot(T - Y_normal))
    Y_normal = sigmoid(Xb.dot(w_normal))


print("Final w_normal:", w_normal)
print("Final w_l2:", w_l2)

