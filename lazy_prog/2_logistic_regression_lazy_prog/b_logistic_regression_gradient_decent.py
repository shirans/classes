import matplotlib.pyplot as plt
import numpy as np

from lazy_prog.common.common import sigmoid, cross_entropy, create_data_2_gaussian_clouds

N = 100
D = 2

Xb, T = create_data_2_gaussian_clouds(N, D)

w = np.random.randn(D + 1)
z = Xb.dot(w)
Y = sigmoid(z)

learning_rate = 0.1
plt.scatter(Xb[:, 1], Xb[:, 2], c=T, s=100, alpha=0.5)
for i in range(100000):
    if i % 10000 == 0:
        print("entropy for step ", i, " : ", cross_entropy(T, Y))
        print("w at step",i,":", w)

    if i ==0 or i == 1 or i == 10:
        x_axis = np.linspace(-6, 6, 100)
        y_axis = -(w[0] + x_axis * w[1]) / w[2]
        plt.plot(x_axis, y_axis, label="iter = " + str(i))

    # gradient descent weight update.
    w += learning_rate * Xb.T.dot(T - Y)

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

print("Final w:", w)
x_axis = np.linspace(-6, 6, 100)
# Y is not the target, just the second coordinate of the x y plane. So: X0*W[0] + X1*W[1] + X2*W[2] = 0 ,
# X1 -> X, X2 -> Y gives that:
y_axis = -(w[0] + x_axis * w[1]) / w[2]
plt.plot(x_axis, y_axis, label="final")
plt.legend()
plt.show()

