from typing import List

import matplotlib.pyplot as plt
import numpy as np

from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid
from models.neural_network_models.mlp import MLP
from models.neural_network_models.train_model import train_model


def run_training():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    mlp_model = MLP(
        input_dim=784, output_dim=10, hidden_dims=[30],
        activation_functions=[sigmoid, sigmoid],
        init_parameters_sd=1
    )

    learning_rate = 1e-2
    print(mlp_model)
    batch_size = 100
    max_epochs = 2

    training_epoch_num, training_losses, validation_losses, validation_accuracies = \
        train_model(mlp_model, x_train, y_train, lr=learning_rate, batch_size=batch_size, max_epochs=max_epochs,
                    x_val=x_val, y_val=y_val)
    plot_losses_during_training(training_epoch_num, training_losses, validation_losses)
    plot_accuracies(training_epoch_num, validation_accuracies)


def plot_losses_during_training(epoch_num: int, training_losses: List[float], validation_losses: List[float]):
    epochs = np.arange(1, epoch_num + 1)
    plt.plot(epochs, training_losses, '*--', c='blue', label='zb. treningowy')
    if len(validation_losses) != 0:
        plt.plot(epochs, validation_losses, '*--', c='orange', label='zb. walidacyjny')
    plt.xlabel('Epoka')
    plt.ylabel('Średnia funkcja straty')
    plt.legend()
    plt.grid(axis='y')
    plt.show()


def plot_accuracies(epoch_num: int, validation_accuracies: List[float]):
    epochs = np.arange(1, epoch_num + 1)
    plt.plot(epochs, validation_accuracies, '*--', label='dokładność')
    plt.legend()
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.grid(axis='y')
    plt.show()


if __name__ == '__main__':
    run_training()
