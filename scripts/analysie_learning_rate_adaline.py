from typing import List

import matplotlib.pyplot as plt
import numpy as np

from dataset.and_dataset import get_dataset
from functions.tran_model import train_model
from models.adaline import Adaline


def analyze_mse_for_different_learning_rates():
    weight_limit = 1.0
    error_margin = 0.1
    max_epoch = 100

    powers = np.arange(-2, -6, -1)
    l_rates = np.concatenate([np.power([10.] * len(powers), powers),
                              5 * np.power([10.] * len(powers[1:]), powers[1:])])
    l_rates = np.sort(l_rates)
    dataset = get_dataset(noise_data_number=5)
    mse_for_lr = []
    epochs_num_for_lr = []
    for lr in l_rates:
        adaline = Adaline(2, weight_limit=weight_limit)
        epoch_num, mean_squared_errors = train_model(adaline, dataset, lr,
                                                     error_margin=error_margin,
                                                     max_epoch=max_epoch,
                                                     unipolar=False,
                                                     plot_epoch=False)
        epochs_num_for_lr.append(epoch_num)
        mse_for_lr.append(mean_squared_errors)

    plot_mse_for_different_lr(l_rates, epochs_num_for_lr, mse_for_lr)


def analyze_epochs_number_for_various_learning_rate_and_init_weights(simulations_num: int = 10):
    max_epoch = 1000
    error_margin = 0.2

    powers = np.arange(-2, -6, -1)
    l_rates = np.concatenate([np.power([10.] * len(powers), powers),
                              5 * np.power([10.] * len(powers[1:]), powers[1:])])
    l_rates = np.sort(l_rates)
    weight_limit = np.array([0.1, 0.2, 0.5, 1.0])

    dataset = get_dataset(noise_data_number=5)
    avg_epochs_numbers = []
    for w_limit in weight_limit:
        avg_epochs_numbers_for_weight = []
        for lr in l_rates:
            lr_epochs_num = []
            for _ in range(simulations_num):
                perceptron = Adaline(2, weight_limit=w_limit)
                epoch_num, mse = train_model(perceptron, dataset, lr, max_epoch=max_epoch, verbose=False,
                                             error_margin=error_margin)
                print(f'MSE: {mse[-1]}')
                lr_epochs_num.append(epoch_num)

            avg_epochs_numbers_for_weight.append(np.mean(lr_epochs_num))

        avg_epochs_numbers.append(avg_epochs_numbers_for_weight)

    avg_epochs_numbers = np.array(avg_epochs_numbers)
    plot_results(l_rates, weight_limit, avg_epochs_numbers)


def plot_mse_for_different_lr(learning_rates: np.ndarray, epochs_num_for_lr: List[int], mse_for_lr: List[List[float]]):
    for lr, epoch_num, mean_squared_errors in zip(learning_rates, epochs_num_for_lr, mse_for_lr):
        plt.plot(np.arange(1, epoch_num + 1), mean_squared_errors, 'o-', label=r'$lr={0}$'.format(lr))

    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    plt.show()


def plot_results(learning_rates: np.ndarray, weight_limit: np.ndarray, epoch_num: np.ndarray):
    for weight, epoch_num_for_weight in zip(weight_limit, epoch_num):
        plt.plot(learning_rates, epoch_num_for_weight, '*--',
                 label=r'$w_{{\mathrm{{pocz}}}}\in(${0}$,${1}$)$'.format(-weight, weight))

    plt.legend()
    plt.grid()
    plt.semilogx()
    plt.xlabel('Współczynnik uczenia')
    plt.ylabel('Liczba epok')
    plt.show()


if __name__ == '__main__':
    analyze_mse_for_different_learning_rates()
    analyze_epochs_number_for_various_learning_rate_and_init_weights(25)
