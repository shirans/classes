import numpy as np

from lazy_prog.common.common import create_data_2_gaussian_clouds, print_cross_entropy_error_for_w, \
    visualize_two_classes_separation

N = 100
D = 2

Xb, T = create_data_2_gaussian_clouds(N, D)

# W(D+1,1)
w_random = np.random.randn(D + 1)

# calculate the cross-entropy error of a randomly selected W:
print_cross_entropy_error_for_w(Xb, w_random, T)
# try it with our closed-form solution
w_closed_solution = np.array([0, 4, 4])

print_cross_entropy_error_for_w(Xb, w_closed_solution, T)


# the best separator
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
visualize_two_classes_separation(Xb, T,x_axis, y_axis, "optimal separator")

# Log likelyhood: given P(y=1 | X) = sigmoid(WtX) = Y (denote sig(...) as Y)
# L of (X1,...Xn) is the multiplication of them, i.e. L (X1,...Xn) = f(X1)*...*f(Xn)
# in our case, likelyhood can be represented like this, because f(x) is either 0 or 1.
# L = 1...N multiplication of Yn^tn*(1-Yn)^(1-tn)
# The bottom line is the minimizing the error is like maximizing the log likelyhood
