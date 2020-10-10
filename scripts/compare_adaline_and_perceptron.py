from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from dataset.and_dataset import get_dataset
from functions.activation_functions import bipolar_activation
from functions.train_model import train_model
from models.adaline import Adaline
from models.perceptron import Perceptron


def compare_models():
    simulations_number = 30

    weight_limit = 0.4
    error_margin = 0.3

    dataset = get_dataset(noise_data_number=0, unipolar=False)

    powers = np.arange(-1, -4, -1)
    learning_rates = np.concatenate([np.power([10.] * len(powers), powers),
                                     5 * np.power([10.] * len(powers[1:]), powers[1:])])
    learning_rates = np.sort(learning_rates)[1:]
    perceptron_avg_epochs_numbers = []
    adaline_avg_epochs_numbers = []

    for lr in learning_rates:
        lr_epochs_num = []
        for _ in range(simulations_number):
            perceptron = Perceptron(2, weight_limit=weight_limit, activation_fn=bipolar_activation)
            epoch_num, _ = train_model(perceptron, dataset, lr, verbose=False)
            lr_epochs_num.append(epoch_num)

        perceptron_avg_epochs_numbers.append(lr_epochs_num)

        lr_epochs_num = []
        for _ in range(simulations_number):
            adaline = Adaline(2, weight_limit=weight_limit)
            epoch_num, mse = train_model(adaline, dataset, lr, verbose=False, error_margin=error_margin)
            print(f'MSE: {mse[-1]}')
            lr_epochs_num.append(epoch_num)

        adaline_avg_epochs_numbers.append(lr_epochs_num)

    plot_result(learning_rates, perceptron_avg_epochs_numbers, adaline_avg_epochs_numbers)


def plot_result(learning_rates: np.ndarray,
                perceptron_avg_epochs_numbers: List[List[float]],
                adaline_avg_epochs_numbers: List[List[float]]) -> None:
    step = 1.5
    shift_rate = 0.2
    positions = np.arange(0, step * len(learning_rates), step)
    perceptron_positions = positions - shift_rate * step
    adaline_positions = positions + shift_rate * step

    perceptron_color = 'lightgreen'
    adaline_color = 'lightblue'

    perceptron_boxplots = plt.boxplot(perceptron_avg_epochs_numbers, positions=perceptron_positions, patch_artist=True)
    adaline_boxplots = plt.boxplot(adaline_avg_epochs_numbers, positions=adaline_positions, patch_artist=True)

    set_box_color(perceptron_boxplots, color=perceptron_color)
    set_box_color(adaline_boxplots, color=adaline_color)

    plt.xticks(positions, learning_rates)

    legend_elements = [mpatches.Patch(facecolor=perceptron_color, edgecolor='k',
                                      label='Perceptron'),
                       mpatches.Patch(facecolor=adaline_color, edgecolor='k',
                                      label='Adaline')]
    plt.legend(handles=legend_elements)
    plt.grid(axis='y')
    plt.xlabel('Współczynnik uczenia')
    plt.ylabel('Liczba epok')
    plt.show()


def set_box_color(bp, color: str) -> None:
    plt.setp(bp['boxes'], facecolor=color)


if __name__ == '__main__':
    compare_models()
