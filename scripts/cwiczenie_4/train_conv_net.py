import math

import numpy as np

from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid, relu
from models.neural_network_models.convolutional_net import ConvolutionalNet
from models.neural_network_models.train_model import train_model
from optimizers.adam import Adam
from optimizers.sgd import SGD
from weight_initilization.he_initializer import HeInitializer


def run_training():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    x_train = np.array([np.reshape(x, (28, 28)) for x in x_train])
    x_val = np.array([np.reshape(x, (28, 28)) for x in x_val])
    x_test = np.array([np.reshape(x, (28, 28)) for x in x_test])

    x_train = x_train[:2000]
    y_train = y_train[:2000]

    x_val = x_val[:200]
    y_val = y_val[:200]

    learning_rate = 5e-3
    batch_size = 50
    max_epochs = 8
    kernel_number = 4
    kernel_size = 5
    padding = 1
    stride = 1
    max_pooling = True

    output_feature_map_dim = math.floor((28 - kernel_size + 2 * padding) / stride + 1)
    if max_pooling:
        output_feature_map_dim = math.floor(output_feature_map_dim / 2)

    conv_net = ConvolutionalNet(
        input_dim=(28, 28),
        kernel_number=kernel_number,
        kernel_size=kernel_size,
        fc_input_dim=kernel_number * output_feature_map_dim ** 2, output_dim=10, hidden_dims=[128],
        activation_functions=[relu],
        optimizer=Adam(learning_rate=learning_rate),
        initializer=HeInitializer()
    )

    print(conv_net)

    index = 1
    x, y = x_test[index, :], y_test[index, :]
    y_hat = conv_net(x)
    print(f'y_real:\n{y}')
    print('Before learning')
    print(f'\ny_hat:\n{y_hat}')

    train_model(conv_net, x_train, y_train, batch_size=batch_size, max_epochs=max_epochs,
                x_val=x_val, y_val=y_val, plot=True)

    y_hat = conv_net(x)
    print(f'y_real:\n{y}')
    print('After learning')
    print(f'\ny_hat:\n{y_hat}')


if __name__ == '__main__':
    run_training()
