import numpy as np
import cv2
import multiprocessing
import pandas as pd
import time
from sklearn.decomposition import PCA

def Gabor(image_set, name):
    num = 1
    kernels = []
    queue = multiprocessing.Queue(maxsize=0)
    phi = 0
    for theta in range(8):
        theta = theta / 4. * np.pi
        for sigma in (1, 3, 5, 7):
            for lamda in np.arange(np.pi / 4, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    for ksize in (3, 5):
                        save_name = name + str(num)
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                        kernels.append(kernel)
                        queue.put((apply_kernel, image_set, kernel, save_name))
                        num += 1
    return queue


def apply_kernel(image_set, kernel):
    filtered_images = []
    for image in image_set:
        filtered_images.append(cv2.filter2D(image, cv2.CV_8U, kernel))
    return filtered_images


if __name__ == '__main__':
    from utils import *
    from save_data import multiprocess_starter
    import knn

    start_path = './data/fashion/'
    end_path = './data/filters/train/'
    # multiprocess_starter(Gabor(load_mnist(start_path, 'train')[0], 'train'), end_path, 8)

    start = time.time()
    x_train, y_train = load_data('./data/filters/train', 'train', 100)
    x_test, y_test = load_data('./data/filters/test', 'test', 100)
    model = knn.model(3, x_train, y_train)
    x = np.reshape(x_train, (-1, 784))
    pca = PCA(n_components=2)
    x = pca.fit_transform(x)
    model2 = knn.model(3, x, y_train)
    # score = knn.score(model, x_test, y_test)
    knn.draw_plot(model2, x_train, y_train)
    print(time.time() - start)
    # print(score)
