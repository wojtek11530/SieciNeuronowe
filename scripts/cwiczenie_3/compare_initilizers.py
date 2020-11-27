import pickle as pkl
from datetime import datetime
import multiprocessing
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid, relu
from models.neural_network_models.mlp import MLP
from models.neural_network_models.train_model import train_model
from optimizers.sgd import SGD
from weight_initilization.basa_initializer import Initializer
from weight_initilization.he_initializer import HeInitializer
from weight_initilization.normal_distr_initilizer import NormalDistributionInitializer
from weight_initilization.xavier_initializer import XavierInitializer

simulation_number = 10
act_function = sigmoid
max_epochs = 7
batch_size = 50
hidden_dims = [100]
weight_sd = 1.0
optimizer = SGD(learning_rate=1e-2)


def analyze_initializers():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    initializers_names = ['Normal', 'Xavier', 'He']
    data_dictionary = {}

    processes = min(len(initializers_names), multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.starmap(get_results_for_initializer,
                               [(initializer_name, x_train, x_val, y_train, y_val)
                                for initializer_name in initializers_names])
        for name, res in zip(initializers_names, results):
            data_dictionary[name] = res

    file_name = f'initializers_analysis_data_{initializers_names}_sigmoid' \
                f'_{datetime.now().strftime("%m-%d-%Y_%H.%M")}.pkl'
    with open(file_name, 'wb') as handle:
        pkl.dump(data_dictionary, handle, protocol=pkl.HIGHEST_PROTOCOL)

    plot_losses_results(data_dictionary)
    plot_accuracies_results(data_dictionary)
    plot_accuracies_boxplot(data_dictionary)


def get_results_for_initializer(initializer_name: str, x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray,
                                y_val: np.ndarray) -> Dict:
    epochs_num = []
    training_losses = []
    validation_losses = []
    validation_accuracies = []

    for i in range(simulation_number):
        print(f'\n{datetime.now().strftime("%m-%d-%Y_%H.%M")} Initializer : {initializer_name}' +
              f' simulation {i + 1}/{simulation_number}')

        initializer = _get_initializer_by_name(initializer_name)

        mlp_model = MLP(
            input_dim=784, output_dim=10, hidden_dims=hidden_dims,
            activation_functions=[act_function],
            optimizer=optimizer,
            initializer=initializer
        )

        sim_overall_epoch_num, sim_training_losses, sim_validation_losses, sim_validation_accuracies = \
            train_model(
                mlp_model, x_train, y_train,
                batch_size=batch_size, max_epochs=max_epochs,
                x_val=x_val, y_val=y_val, plot=False
            )

        epochs_num.append(sim_overall_epoch_num)
        training_losses.append(sim_training_losses)
        validation_losses.append(sim_validation_losses)
        validation_accuracies.append(sim_validation_accuracies)

    return {'epochs': epochs_num, 'train_losses': training_losses,
            'val_losses': validation_losses, 'val_acc': validation_accuracies,
            'optimizer': optimizer}


def _get_initializer_by_name(initializer_name: str) -> Initializer:
    if initializer_name == 'Xavier':
        initializer = XavierInitializer()
    elif initializer_name == 'He':
        initializer = HeInitializer()
    else:
        initializer = NormalDistributionInitializer(sd=weight_sd)
    return initializer


def analyze_initializers_from_file():
    file_name = \
        "initializers_analysis_data_['Normal', 'Xavier', 'He']_relu_11-25-2020_12.51.pkl"
    with open(file_name, 'rb') as handle:
        training_data_dictionary = pkl.load(handle)
        plot_losses_results(training_data_dictionary)
        plot_accuracies_results(training_data_dictionary)
        plot_accuracies_boxplot(training_data_dictionary)


def plot_losses_results(training_data_dictionary: Dict[str, Dict]):
    plt.figure(figsize=(5, 6))
    for initializer, values_dict in training_data_dictionary.items():
        epoch_num = values_dict['epochs'][0]
        epochs = np.arange(1, epoch_num + 1)
        avg_training_losses = np.mean(np.array(values_dict['train_losses']), axis=0)
        avg_val_losses = np.mean(np.array(values_dict['val_losses']), axis=0)

        plt.plot(epochs, avg_training_losses, '*:', ms=6,
                 label=f'zb. tren., {initializer}')
        plt.plot(epochs, avg_val_losses, 'o--', ms=6, c=plt.gca().lines[-1].get_color(),
                 label=f'zb. wal.,  {initializer}')

    plt.xlabel('Epoka')
    plt.ylabel('Średnia funkcja straty')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_accuracies_results(training_data_dictionary: Dict[str, Dict]):
    plt.figure(figsize=(5, 6))
    for initializer, values_dict in training_data_dictionary.items():
        epoch_num = values_dict['epochs'][0]
        epochs = np.arange(1, epoch_num + 1)
        avg_val_acc = np.mean(np.array(values_dict['val_acc']), axis=0)
        plt.plot(epochs, avg_val_acc, '*--', ms=6,
                 label=f'zb. tren., {initializer}')

    plt.xlabel('Epoka')
    plt.ylabel('Średnia dokładność')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_accuracies_boxplot(data_dictionary: Dict[str, Dict]):
    last_epoch_accuracies = [np.array(values_dict['val_acc'])[:, -1] for values_dict in
                             data_dictionary.values()]
    optimizers = data_dictionary.keys()

    plt.boxplot(last_epoch_accuracies, labels=optimizers)
    plt.ylabel('Dokładność na zb. wal. w ostatniej epoce')
    plt.xlabel('Inicjalizator')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # analyze_initializers()
    analyze_initializers_from_file()
