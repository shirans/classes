# shows how linear regression analysis can be applied to moore's law
#
# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python
# transistor count from: https://en.wikipedia.org/wiki/Transistor_count

from __future__ import print_function, division
# from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

# some numbers show up as 1,170,000,000 (commas)
# some numbers have references in square brackets after them
non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    X.append(x)
    Y.append(y)


X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()


Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

print("a:{}".format(a))
print("b:{}".format(b))
# let's calculate the predicted Y
Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()