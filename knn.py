from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def model(k, x_train, y_train):
    knn = KNeighborsClassifier(k)
    x_train = np.reshape(x_train, (-1, 784))
    knn.fit(x_train, y_train)
    return knn


def score(knn, x_test, y_test):
    x_test = np.reshape(x_test, (-1, 784))
    return knn.score(x_test, y_test)

