from typing import List

import matplotlib.pyplot as plt
import numpy as np

from dataset.and_dataset import get_dataset
from functions.activation_functions import unipolar_activation
from models.simple_models.train_model import train_model
from models.simple_models.perceptron import Perceptron


def analyze(simulations_num: int = 10):
    lr = 0.01
    weight_limit = np.array([0.1, 0.3, 0.5, 0.8, 1.0])

    dataset = get_dataset(noise_data_number=5, unipolar=True)
    epochs_numbers = []
    for w_limit in weight_limit:

        w_epochs_num = []
        for _ in range(simulations_num):
            perceptron = Perceptron(2, weight_limit=w_limit, activation_fn=unipolar_activation)
            epoch_num, _ = train_model(perceptron, dataset, lr, verbose=False)
            w_epochs_num.append(epoch_num)

        epochs_numbers.append(w_epochs_num)

    plot_results(weight_limit, epochs_numbers)


def plot_results(weight_limit: np.ndarray,
                 epochs_numbers: List[List[float]]) -> None:
    plt.boxplot(epochs_numbers)

    positions = list(1 + np.arange(len(weight_limit)))
    plt.xticks(positions,
               [r'$(${0}$,${1}$)$'.format(-weigth, weigth) for weigth in weight_limit])

    plt.grid(axis='y')
    plt.xlabel(r'Wagi początkowe - przedział')
    plt.ylabel('Liczba epok')
    plt.show()


if __name__ == '__main__':
    analyze()
