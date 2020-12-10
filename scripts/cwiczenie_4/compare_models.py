import math
import pickle as pkl
from datetime import datetime
import multiprocessing
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid, relu
from models.neural_network_models.convolutional_net import ConvolutionalNet
from models.neural_network_models.mlp import MLP
from models.neural_network_models.train_model import train_model
from optimizers.adam import Adam
from weight_initilization.he_initializer import HeInitializer

train_data_num = 3000
val_data_num = 300

simulation_number = 4
max_epochs = 5
batch_size = 50
learning_rate = 5e-3

mlp_hidden_dims = [100]

fc_act_function = relu

kernel_number = 4
kernel_sizes = [3, 5, 7]
padding = 1
stride = 1
max_pooling = True


def analyze_models():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    x_train = x_train[:train_data_num]
    y_train = y_train[:train_data_num]

    x_val = x_val[:val_data_num]
    y_val = y_val[:val_data_num]

    data_dictionary = {}

    processes = min(simulation_number, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=processes) as pool:

        epochs_num = []
        training_losses = []
        validation_losses = []
        validation_accuracies = []

        results = pool.starmap(get_results_for_mlp, [(x_train, x_val, y_train, y_val, i)
                                                     for i in range(simulation_number)])

        for sim_overall_epoch_num, sim_training_losses, sim_validation_losses, sim_validation_accuracies in results:
            epochs_num.append(sim_overall_epoch_num)
            training_losses.append(sim_training_losses)
            validation_losses.append(sim_validation_losses)
            validation_accuracies.append(sim_validation_accuracies)

        data_dictionary[f'MLP'] = \
            {'epochs': epochs_num, 'train_losses': training_losses,
             'val_losses': validation_losses, 'val_acc': validation_accuracies}

        for k in kernel_sizes:
            epochs_num = []
            training_losses = []
            validation_losses = []
            validation_accuracies = []

            results = pool.starmap(get_results_for_cnn, [(x_train, x_val, y_train, y_val, k, i)
                                                         for i in range(simulation_number)])

            for sim_overall_epoch_num, sim_training_losses, sim_validation_losses, sim_validation_accuracies in results:
                epochs_num.append(sim_overall_epoch_num)
                training_losses.append(sim_training_losses)
                validation_losses.append(sim_validation_losses)
                validation_accuracies.append(sim_validation_accuracies)

            data_dictionary[f'CNN k={k}'] = \
                {'epochs': epochs_num, 'train_losses': training_losses,
                 'val_losses': validation_losses, 'val_acc': validation_accuracies}

        file_name = f'models_analysis_data' \
                    f'_{datetime.now().strftime("%m-%d-%Y_%H.%M")}.pkl'
        with open(file_name, 'wb') as handle:
            pkl.dump(data_dictionary, handle, protocol=pkl.HIGHEST_PROTOCOL)

        plot_losses_results(data_dictionary)
        plot_accuracies_results(data_dictionary)
        plot_accuracies_boxplot(data_dictionary)


def get_results_for_cnn(x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray,
                        y_val: np.ndarray, kernel_size: int, simulation_i: int) -> Tuple:
    print(f'\n{datetime.now().strftime("%m-%d-%Y_%H.%M")} Model: CNN k={kernel_size}' +
          f' simulation {simulation_i + 1}/{simulation_number}')

    x_train = np.array([np.reshape(x, (28, 28)) for x in x_train])
    x_val = np.array([np.reshape(x, (28, 28)) for x in x_val])

    output_feature_map_dim = math.floor((28 - kernel_size + 2 * padding) / stride + 1)
    if max_pooling:
        output_feature_map_dim = math.floor(output_feature_map_dim / 2)

    conv_net = ConvolutionalNet(
        input_dim=(28, 28),
        kernel_number=kernel_number,
        kernel_size=kernel_size,
        fc_input_dim=kernel_number * output_feature_map_dim ** 2, output_dim=10, hidden_dims=[128],
        activation_functions=[fc_act_function],
        optimizer=Adam(learning_rate=learning_rate),
        initializer=HeInitializer()
    )

    sim_overall_epoch_num, sim_training_losses, sim_validation_losses, sim_validation_accuracies = \
        train_model(
            conv_net, x_train, y_train,
            batch_size=batch_size, max_epochs=max_epochs,
            x_val=x_val, y_val=y_val, plot=False
        )

    return sim_overall_epoch_num, sim_training_losses, sim_validation_losses, sim_validation_accuracies


def get_results_for_mlp(x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray,
                        y_val: np.ndarray, simulation_i: int) -> Tuple:
    print(f'\n{datetime.now().strftime("%m-%d-%Y_%H.%M")} Model: MLP' +
          f' simulation {simulation_i + 1}/{simulation_number}')

    mlp_model = MLP(
        input_dim=784, output_dim=10, hidden_dims=mlp_hidden_dims,
        activation_functions=[fc_act_function],
        optimizer=Adam(learning_rate=learning_rate),
        initializer=HeInitializer()
    )

    sim_overall_epoch_num, sim_training_losses, sim_validation_losses, sim_validation_accuracies = \
        train_model(
            mlp_model, x_train, y_train,
            batch_size=batch_size, max_epochs=max_epochs,
            x_val=x_val, y_val=y_val, plot=False
        )

    return sim_overall_epoch_num, sim_training_losses, sim_validation_losses, sim_validation_accuracies


def analyze_initializers_from_file():
    file_name = \
        "models_analysis_data.pkl"
    with open(file_name, 'rb') as handle:
        training_data_dictionary = pkl.load(handle)
        analyze_cnn_models(training_data_dictionary)
        compare_cnn_and_mlp_models(training_data_dictionary)


def analyze_cnn_models(training_data_dictionary: Dict[str, Dict]):
    training_data_dictionary = {key: val for key, val in training_data_dictionary.items() if key != 'MLP'}
    plot_losses_results(training_data_dictionary)
    plot_accuracies_results(training_data_dictionary)
    plot_accuracies_boxplot(training_data_dictionary)


def compare_cnn_and_mlp_models(training_data_dictionary: Dict[str, Dict]):
    excluded = ['CNN k=3', 'CNN k=7']
    training_data_dictionary = {key: val for key, val in training_data_dictionary.items()
                                if key not in excluded}
    plot_losses_results(training_data_dictionary)
    plot_accuracies_results(training_data_dictionary)
    plot_accuracies_boxplot(training_data_dictionary)


def plot_losses_results(training_data_dictionary: Dict[str, Dict]):
    plt.figure(figsize=(5, 6))
    for model, values_dict in training_data_dictionary.items():
        epoch_num = values_dict['epochs'][0]
        epochs = np.arange(1, epoch_num + 1)
        avg_training_losses = np.mean(np.array(values_dict['train_losses']), axis=0)
        avg_val_losses = np.mean(np.array(values_dict['val_losses']), axis=0)

        plt.plot(epochs, avg_training_losses, '*:', ms=6,
                 label=f'zb. tren., {model}')
        plt.plot(epochs, avg_val_losses, 'o--', ms=6, c=plt.gca().lines[-1].get_color(),
                 label=f'zb. wal.,  {model}')

    plt.xlabel('Epoka')
    plt.ylabel('Średnia funkcja straty')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_accuracies_results(training_data_dictionary: Dict[str, Dict]):
    plt.figure(figsize=(5, 6))
    for model, values_dict in training_data_dictionary.items():
        epoch_num = values_dict['epochs'][0]
        epochs = np.arange(1, epoch_num + 1)
        avg_val_acc = np.mean(np.array(values_dict['val_acc']), axis=0)
        plt.plot(epochs, avg_val_acc, '*--', ms=6,
                 label=f'zb. tren., {model}')

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
    plt.xlabel('Model')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # analyze_models()
    analyze_initializers_from_file()
