import matplotlib.pyplot as plt
import numpy as np

from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid, relu
from models.neural_network_models.convolutional_net import ConvolutionalNet
from models.neural_network_models.train_model import train_model
from optimizers.sgd import SGD
from weight_initilization.he_initializer import HeInitializer


def run_training():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    x_train = np.array([np.reshape(x, (28, 28)) for x in x_train])
    x_val = np.array([np.reshape(x, (28, 28)) for x in x_val])
    x_test = np.array([np.reshape(x, (28, 28)) for x in x_test])

    learning_rate = 1e-1
    batch_size = 50
    max_epochs = 8

    conv_net = ConvolutionalNet(
        input_dim=(28, 28),
        fc_input_dim=25088, output_dim=10, hidden_dims=[128],
        activation_functions=[relu],
        optimizer=SGD(learning_rate=learning_rate),
        initializer=HeInitializer()
    )

    print(conv_net)

    limit = 1
    i = 1
    for x, y in zip(x_train, y_train):
        y_hat = conv_net(x)
        print(f'y_real:\n{y}')
        print(f'\ny_hat:\n{y_hat}')
        i += 1
        if i > limit:
            break

    # train_model(mlp_model, x_train, y_train, lr=learning_rate, batch_size=batch_size, max_epochs=max_epochs,
    #             x_val=x_val, y_val=y_val, plot=True)


if __name__ == '__main__':
    run_training()
