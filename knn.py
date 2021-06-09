from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA


def model(k, x_train, y_train):
    knn = KNeighborsClassifier(k)
    x_train = np.reshape(x_train, (-1, 784))
    knn.fit(x_train, y_train)
    return knn


def score(knn, x_test, y_test):
    x_test = np.reshape(x_test, (-1, 784))
    return knn.score(x_test, y_test)


def draw_plot(knn, x, y):
    x = np.reshape(x, (-1, 784))
    pca = PCA(n_components=2)
    x2 = pca.fit_transform(x)
    plot_decision_regions(x2, y, clf=knn, legend=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Knn with K=' + str(knn.n_neighbors))
    plt.show()
