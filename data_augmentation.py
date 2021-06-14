import numpy as np
from utils import *


def fashion_pixel_prep():
    x_train, y_train = load_mnist('./data/fashion/', 'train')
    x_test, y_test = load_mnist('./data/fashion/', 't10k')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test


def add_salt_pepper_noise(data, amount, salt_vs_pepper):
    # Need to produce a copy as to not modify the original image
    data_copy = data.copy()
    row, col = data_copy[0].shape
    num_salt = np.ceil(amount * data_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * data_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in data_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1]] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1]] = 0
    return data_copy
