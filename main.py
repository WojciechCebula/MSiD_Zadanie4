import time
from utils import *
from save_data import multiprocess_starter
from filters import *
from tqdm import tqdm
import knn


def filters_generator(data_type):
    start_path = './data/fashion/'
    end_path = f'./data/filters/{data_type}/'
    multiprocess_starter(gabor(load_mnist(start_path, data_type)[0], data_type), end_path, 8)


def knn_model(index, k):
    x_train, y_train = load_data('./data/filters/train', 'train', index)
    x_test, y_test = load_data('./data/filters/test', 'test', index)
    model = knn.model(k, x_train, y_train)
    return knn.score(model, x_test, y_test)


def all_knn_models(start, end, k_array):
    scores = []
    i = start
    for i in tqdm(range(end)):
        for k in k_array:
            scores.append((knn_model(start, k), i, k))
    return scores


if __name__ == '__main__':
    print(all_knn_models(1, 3, [3, 5]))
