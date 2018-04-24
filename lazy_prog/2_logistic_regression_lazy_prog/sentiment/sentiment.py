from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from lazy_prog.common.common import sigmoid, cross_entropy, cross_entropy_numpysum

positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), "html5lib")
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), "html5lib")
negative_reviews = negative_reviews.findAll('review_text')

wordnet_lemmatizer = WordNetLemmatizer()
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
stopwords = stopwords.words('english')

if (len(positive_reviews) != len(negative_reviews)):
    # if netative > positive, can either over sample negative or take len(negative) of positive
    print("datasets are not balanced")


def tokinizer(s):
    s_old = s
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]  # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # put words into base form
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    return tokens
    # print("before:" + s_old)
    # print("after:", ','.join(tokens))


word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []


def add_works_to_index(reviews, current_index, tokenized):
    for review in reviews:
        tokens = tokinizer(review.text)
        tokenized.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
    return current_index


current_index = add_works_to_index(positive_reviews, current_index, positive_tokenized)
current_index = add_works_to_index(negative_reviews, current_index, negative_tokenized)


def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)  # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()  # normalize it before setting label
    x[-1] = label
    return x


N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_index_map) + 1))

i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i, :] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i, :] = xy
    i += 1

np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

# last 100 rows will be test
Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]
Xtest = X[-100:, ]
Ytest = Y[-100:, ]


def logistic_regression(xtrain, ytrain):
    alpha = 0.01
    [N, D] = xtrain.shape
    w = np.random.random(D)
    b = 0
    for i in range(0, 100):
        ytag = sigmoid(xtrain.dot(w) + b)
        if i % 10 == 0:
            print("current cross entropy:", cross_entropy(ytrain, ytag))
            print("current success rate traijn:", np.mean(np.mean(np.round(ytag) - ytrain)))
            ytagTest = sigmoid(Xtest.dot(w))
            print("current success rate test:", np.mean(np.mean(np.round(ytagTest) - Ytest)))
        w -= alpha * xtrain.T.dot(ytag - ytrain)
        b -= alpha * (ytag - ytrain).sum()
    return w


# N = Xtrain.shape[0]
# zeros = np.zeros((N, 1))
# Xb = np.concatenate((zeros, Xtrain), axis=1)

print("Xtrain shape", Xtrain.shape)
print("Ytrain shape", Ytrain.shape)
print("Xtrain shape", Xtest.shape)
print("Ytest shape", Ytest.shape)
w = logistic_regression(Xtrain, Ytrain)
print("w shape", w.shape)
res = sigmoid(Xtest.dot(w))
print("cross entropy:", cross_entropy_numpysum(Ytest, res))
print("classification rate:", np.mean(Ytest == np.round(res)))
