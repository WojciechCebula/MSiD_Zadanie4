import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks
import data_augmentation

NUMBER_OF_OUTPUTS = 10


def simple_neural_network(number_of_hidden_layers, number_of_neurons):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(28, 28)))
    model.add(layers.Flatten())
    for _ in range(number_of_hidden_layers):
        model.add(layers.Dense(number_of_neurons, activation='relu'))
    model.add(layers.Dense(NUMBER_OF_OUTPUTS, activation='softmax'))
    return model


def conv_neural_network_1():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUMBER_OF_OUTPUTS, activation='softmax'))
    return model


def train_conv(model, optimizer,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 salt_and_pepper=False, amount=0.01, salt_vs_pepper=0.5, probability=1):
    x_train, y_train, x_test, y_test = data_augmentation.fashion_pixel_prep()
    if salt_and_pepper:
        x_train = data_augmentation.add_salt_pepper_noise(x_train, amount, salt_vs_pepper, probability)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose= 1, batch_size=512,
              callbacks=[callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5),
                         callbacks.EarlyStopping(monitor='val_accuracy', patience=6, min_delta=1e-4)])


def train_normal(number_of_hidden_layers, number_of_neurons,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-08),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 salt_and_pepper=False, amount=0.01, salt_vs_pepper=0.5, probability=1,
                 filters=False, filter=1):

    model = simple_neural_network(number_of_hidden_layers, number_of_neurons)
    x_train, y_train, x_test, y_test = data_augmentation.fashion_pixel_prep(filters, filter)
    if salt_and_pepper:
        x_train = data_augmentation.add_salt_pepper_noise(x_train, amount, salt_vs_pepper, probability)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose= 1, batch_size=512,
              callbacks=[callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5),
                         callbacks.EarlyStopping(monitor='val_accuracy', patience=6, min_delta=1e-4)])

    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return test_acc


def simple_networks_test():
    layers = [1, 3, 5, 10]
    neurons = [64, 128, 256, 512, 1024]
    results = []
    for layer in layers:
        for neuron in neurons:
            results.append((train_normal(layer, neuron, salt_and_pepper=False, filters=True, filter=195), layer, neuron, 0))
            results.append((train_normal(layer, neuron, salt_and_pepper=True, filters=True, filter=195), layer, neuron, 1))
    results.sort(reverse=True)
    for result in results:
        print(result)
