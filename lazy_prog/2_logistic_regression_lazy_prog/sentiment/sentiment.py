from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
from sklearn.model_selection import KFold

from models import sklearn_l1, sklearn, course_logistic, sklearn_l1_then_l2, sklearn_l2
import logging.config


LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(levelname)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console'],
        },
    },
}


logging.config.dictConfig(LOGGING)

logging.getLogger("sklearn").setLevel(logging.DEBUG)
logging.getLogger(__name__).debug('This is a debug message')

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
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]  # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # put words into base form
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    return tokens


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

# last 100 rows will be test
X = data[:, :-1]
Y = data[:, -1]
Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]
Xtest = X[-100:, ]
Ytest = Y[-100:, ]

#####################################################################################################################


total_start = time.time()
kf = KFold(n_splits=10)
index = 0
scores = []
weight_word_index = {}
for train, test in kf.split(data):
    start = time.time()
    print("running fold:", index)
    print("data shapes: %s %s" % (train.shape, test.shape))
    Xtrain = data[train][:, :-1]
    Ytrain = data[train][:,-1]
    Xtest = data[test][:, :-1]
    Ytest = data[test][:, -1]

    # score = course_logistic(xtrain=Xtrain, ytrain=Ytrain, xtest=Xtest, ytest=Ytest)

    score, weight_word_index = sklearn_l1_then_l2(xtrain=Xtrain, ytrain=Ytrain, xtest=Xtest, ytest=Ytest, word_index_map=word_index_map, weight_word_index=weight_word_index)

    scores.append(score)
    end = time.time()
    print("total time", end - start)
    index = index + 1

#####################################################################################################################
total_end = time.time()
print("weight_word_index: ",weight_word_index)
print("total time", total_end - total_start)
print("scores:", scores)
print("scores avg:", np.average(scores))
