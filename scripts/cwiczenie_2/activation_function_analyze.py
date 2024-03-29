import pickle as pkl
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import relu, sigmoid
from models.neural_network_models.mlp import MLP
from models.neural_network_models.train_model import train_model


def analyze_activation_functions():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    simulation_number = 5

    max_epochs = 7
    batch_size = 50
    weight_sd = 1.0
    learning_rate = 1e-1

    act_functions = [sigmoid, relu]
    act_functions_names = ['sigmoid', 'relu']

    training_data_dictionary = {}

    for act_fn, act_fn_name in zip(act_functions, act_functions_names):
        epochs_num = []
        training_losses = []
        validation_losses = []
        validation_accuracies = []

        for i in range(simulation_number):
            print(f'\nActivation function : {act_fn_name}, simulation {i + 1}/{simulation_number}')
            mlp_model = MLP(
                input_dim=784, output_dim=10, hidden_dims=[30],
                activation_functions=[act_fn],
                init_parameters_sd=weight_sd,
                optimizer=SGD(learning_rate=learning_rate)
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

        training_data_dictionary[act_fn_name] = {'epochs': epochs_num, 'train_losses': training_losses,
                                                 'val_losses': validation_losses, 'val_acc': validation_accuracies}

    file_name = f'act_functions_analysis_data_{act_functions_names}_{datetime.now().strftime("%m-%d-%Y_%H.%M")}.pkl'
    with open(file_name, 'wb') as handle:
        pkl.dump(training_data_dictionary, handle, protocol=pkl.HIGHEST_PROTOCOL)

    plot_losses_results(training_data_dictionary)
    plot_accuracies_results(training_data_dictionary)


def analyze_activation_functions_from_file():
    file_name = "act_functions_analysis_data_['sigmoid', 'relu']_10-30-2020_16.59.pkl"
    with open(file_name, 'rb') as handle:
        training_data_dictionary = pkl.load(handle)
        plot_losses_results(training_data_dictionary)
        plot_accuracies_results(training_data_dictionary)
        plot_accuracies_boxplot(training_data_dictionary)


def plot_losses_results(training_data_dictionary: Dict[str, Dict]):
    plt.figure(figsize=(5, 6))
    for act_fn_name, values_dict in training_data_dictionary.items():
        epoch_num = values_dict['epochs'][0]
        epochs = np.arange(1, epoch_num + 1)
        avg_training_losses = np.mean(np.array(values_dict['train_losses']), axis=0)
        avg_val_losses = np.mean(np.array(values_dict['val_losses']), axis=0)

        plt.plot(epochs, avg_training_losses, '*:', ms=6,
                 label=f'zb. tren., {act_fn_name}')
        plt.plot(epochs, avg_val_losses, 'o--', ms=6, c=plt.gca().lines[-1].get_color(),
                 label=f'zb. wal., {act_fn_name}')

    plt.xlabel('Epoka')
    plt.ylabel('Średnia funkcja straty')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_accuracies_results(training_data_dictionary: Dict[str, Dict]):
    plt.figure(figsize=(5, 6))
    for act_fn_name, values_dict in training_data_dictionary.items():
        epoch_num = values_dict['epochs'][0]
        epochs = np.arange(1, epoch_num + 1)
        avg_val_acc = np.mean(np.array(values_dict['val_acc']), axis=0)

        plt.plot(epochs, avg_val_acc, '*--', ms=6, label=f'{act_fn_name}')

    plt.xlabel('Epoka')
    plt.ylabel('Średnia dokładność')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_accuracies_boxplot(training_data_dictionary: Dict[str, Dict]):
    last_epoch_accuracies = [np.array(values_dict['val_acc'])[:, -1] for hidden_neuron_num, values_dict in
                             training_data_dictionary.items()]
    act_functions_names = training_data_dictionary.keys()

    plt.boxplot(last_epoch_accuracies, labels=act_functions_names)
    plt.ylabel('Dokładność na zb. wal. w ostatniej epoce')
    plt.xlabel(r'Funkcja aktywacji')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # analyze_activation_functions()
    analyze_activation_functions_from_file()
