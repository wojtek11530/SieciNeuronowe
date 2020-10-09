import matplotlib.pyplot as plt
import numpy as np

from dataset.and_dataset import get_dataset
from functions.tran_model import train_model
from models.perceptron import Perceptron


def analyze(simulations_num: int = 10):
    powers = np.arange(-1, -5, -1)

    l_rates = np.concatenate([[0.99, 0.9, 0.75, 0.5, 0.25], np.power([10.] * len(powers), powers),
                              5 * np.power([10.] * len(powers[1:]), powers[1:])])
    l_rates = np.sort(l_rates)
    weight_limit = np.array([0.2, 0.5, 1.0])

    dataset = get_dataset(noise_data_number=5)
    avg_epochs_numbers = []
    for w_limit in weight_limit:
        avg_epochs_numbers_for_weight = []
        for lr in l_rates:
            lr_epochs_num = []
            for _ in range(simulations_num):
                perceptron = Perceptron(2, weight_limit=w_limit)
                epoch_num, _ = train_model(perceptron, dataset, lr, verbose=False)
                lr_epochs_num.append(epoch_num)

            avg_epochs_numbers_for_weight.append(np.mean(lr_epochs_num))

        avg_epochs_numbers.append(avg_epochs_numbers_for_weight)

    avg_epochs_numbers = np.array(avg_epochs_numbers)
    plot_results(l_rates, weight_limit, avg_epochs_numbers)


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
    analyze()
