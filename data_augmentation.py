from utils import *


def fashion_pixel_prep(filters=False, filter=1):
    if not filters:
        x_train, y_train = load_mnist('./data/fashion/', 'train')
        x_test, y_test = load_mnist('./data/fashion/', 't10k')
    else:
        x_train, y_train = load_data('./data/filters/train/', 'train', filter)
        x_test, y_test = load_data('./data/filters/test/', 'test', filter)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test


def add_salt_pepper_noise(data, amount, salt_vs_pepper, probability):
    data_copy = data.copy()
    num_salt = np.ceil(amount * data_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * data_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in data_copy:
        if np.random.uniform() < probability:
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
            X_img[coords[0], coords[1]] = 1

            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
            X_img[coords[0], coords[1]] = 0
    return data_copy
