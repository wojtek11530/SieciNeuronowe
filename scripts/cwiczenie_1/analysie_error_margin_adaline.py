from typing import List

import matplotlib.pyplot as plt
import numpy as np

from dataset.and_dataset import get_dataset
from models.simple_models.adaline import Adaline
from models.simple_models.train_model import train_model


def analyze(simulations_num: int = 10):
    weight_limit = 1.0
    learning_rate = 0.01
    error_margins = [0.2, 0.25, 0.3, 0.5, 0.7]
    max_epoch = 40
    dataset = get_dataset(noise_data_number=5, unipolar=False)

    epochs_numbers = []
    for error_margin in error_margins:

        err_marg_epochs_num = []
        for _ in range(simulations_num):
            adaline = Adaline(2, weight_limit=weight_limit, error_margin=error_margin)
            epoch_num, mean_squared_errors = train_model(adaline, dataset, learning_rate,
                                                         max_epoch=max_epoch)
            err_marg_epochs_num.append(epoch_num)

        epochs_numbers.append(err_marg_epochs_num)

    plot_results(error_margins, epochs_numbers)


def plot_results(error_margins: List[float],
                 epochs_numbers: List[List[int]]) -> None:
    plt.boxplot(epochs_numbers)

    positions = list(1 + np.arange(len(error_margins)))
    plt.xticks(positions, error_margins)

    plt.grid(axis='y')
    plt.xlabel(r'Dopuszczalna wartość błedu MSE')
    plt.ylabel('Liczba epok')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    analyze()
