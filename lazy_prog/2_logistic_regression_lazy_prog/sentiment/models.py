from sklearn.linear_model import LogisticRegression
from future.utils import iteritems
import numpy as np

from lazy_prog.common.common import sigmoid, cross_entropy, cross_entropy_numpysum, logistic_regression_with_test, \
    classification_rate, logistic_regression, logistic_regression_l1


def course_logistic(xtrain, ytrain, xtest, ytest):
    w = logistic_regression(xtrain, ytrain, 1000, 0.01)
    print("w shape", w.shape)
    print("W avg size:", w.mean())
    y_tag_test = sigmoid(xtest.dot(w))
    y_tag_train = sigmoid(xtrain.dot(w))
    print("cross entropy train:", cross_entropy(ytrain, y_tag_train))
    print("cross entropy test:", cross_entropy(ytest, y_tag_test))
    print("classification rate train:", classification_rate(y_tag_train, ytrain))
    score = classification_rate(y_tag_test, ytest)
    print("classification rate test:", score)
    return score


def sklearn_l1(xtrain, ytrain, xtest, ytest, word_index_map):
    model = LogisticRegression(penalty='l1')
    return sklean_fit_and_run(model, word_index_map, xtrain, ytrain, xtest, ytest)


def sklearn(xtrain, ytrain, xtest, ytest, word_index_map):
    # model = LogisticRegression(solver='sag')
    model = LogisticRegression()
    return sklean_fit_and_run(model, word_index_map, xtrain, ytrain, xtest, ytest)


def sklearn_l2(xtrain, ytrain, xtest, ytest, word_index_map):
    model = LogisticRegression(penalty='l2')
    return sklean_fit_and_run(model, word_index_map, xtrain, ytrain, xtest, ytest)


def sklean_fit_and_run(model, word_index_map, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    score = model.score(xtest, ytest)
    print("Classification rate:", score)
    threshold = 0.5
    for word, index in iteritems(word_index_map):
        weight = model.coef_[0][index]
        if weight > threshold or weight < -threshold:
            print(word, weight)
    return score


def sklearn_l1_then_l2(xtrain, ytrain, xtest, ytest, word_index_map, weight_word_index):
    model = LogisticRegression(penalty='l1')
    model.fit(xtrain, ytrain)
    score = model.score(xtest, ytest)
    print("score with l1:", score)
    w = model.coef_[0]
    indexed_of_nonzero = np.transpose(np.flatnonzero(w))

    for word, index in iteritems(word_index_map):
        weight = w[index]
        if weight != 0:
            if word not in weight_word_index:
                weight_word_index[word] = 0
            else:
                weight_word_index[word] = weight_word_index[word] + 1
            print(word, weight)

    model = LogisticRegression()
    model.fit(xtrain, ytrain)
    score = model.score(xtest, ytest)
    print("score on all features:", score)
    xtrain = xtrain[:, indexed_of_nonzero]
    xtest = xtest[:, indexed_of_nonzero]

    model = LogisticRegression()
    model.fit(xtrain, ytrain)
    score = model.score(xtest, ytest)
    print("score with l2 on all {} features: {}", indexed_of_nonzero.shape, score)
    return score, weight_word_index
