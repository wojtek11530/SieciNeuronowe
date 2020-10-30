from typing import List

import matplotlib.pyplot as plt
import numpy as np

from dataset.and_dataset import get_dataset
from functions.activation_functions import unipolar_activation
from models.simple_models.perceptron import Perceptron
from models.simple_models.train_model import train_model


def analyze(simulations_num: int = 10):
    powers = np.arange(-1, -5, -1)
    w_limit = 0.5

    l_rates = np.concatenate([[0.9, 0.5, 0.25], np.power([10.] * len(powers), powers),
                              5 * np.power([10.] * len(powers[1:]), powers[1:])])
    l_rates = np.sort(l_rates)[1:]

    dataset = get_dataset(noise_data_number=5, unipolar=True)

    epochs_numbers = []
    for lr in l_rates:

        lr_epochs_num = []
        for _ in range(simulations_num):
            perceptron = Perceptron(2, weight_limit=w_limit, activation_fn=unipolar_activation)
            epoch_num, _ = train_model(perceptron, dataset, lr, verbose=False)
            lr_epochs_num.append(epoch_num)

        epochs_numbers.append(lr_epochs_num)

    plot_results(l_rates, epochs_numbers)


def plot_results(learning_rates: np.ndarray,
                 epochs_numbers: List[List[int]]) -> None:
    plt.boxplot(epochs_numbers)

    positions = list(1 + np.arange(len(learning_rates)))
    plt.xticks(positions, learning_rates)

    plt.grid(axis='y')
    plt.xlabel(r'Współczynnik uczenia $\alpha$')
    plt.ylabel('Liczba epok')
    plt.show()


if __name__ == '__main__':
    analyze()
