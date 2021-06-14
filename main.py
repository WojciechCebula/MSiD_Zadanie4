from save_data import multiprocess_starter
from filters import *
from tqdm import tqdm
from neural_networks import *
from utils import *
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


def all_knn_models(start, end, k):
    scores = []
    for i in tqdm(range(start, end)):
        scores.append((knn_model(i, k), i))
    return scores


def all_k_values(i, k_values):
    scores = []
    for k in k_values:
        scores.append((knn_model(i, k), k))
    return scores


def all_filters_test():
    best_filters = all_knn_models(1, 384, 5)
    best_filters.sort(reverse=True)
    print("BEST FILTERS")
    for model in best_filters:
        print(model)
    best_k = all_k_values(best_filters[0][1], [1, 3, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000])
    best_k.sort(reverse=True)
    print("BEST K-VALUES")
    for model in best_k:
        print(model)


if __name__ == '__main__':
    # train_conv(conv_neural_network_1(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-08, decay=0.0),
    #            salt_and_pepper=True)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-08)
    # optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
    pass
