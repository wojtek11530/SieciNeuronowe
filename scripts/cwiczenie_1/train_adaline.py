import matplotlib.pyplot as plt
import numpy as np

from dataset.and_dataset import get_dataset
from models.simple_models.adaline import Adaline
from models.simple_models.evaluate_model import evaluate_model
from models.simple_models.train_model import train_model


def run_training():
    weight_limit = 1.0
    learning_rate = 0.01
    error_margin = 0.3
    max_epoch = 500
    dataset = get_dataset(noise_data_number=20, unipolar=False)

    adaline = Adaline(2, weight_limit=weight_limit, error_margin=error_margin)
    epoch_num, mean_squared_errors = train_model(adaline, dataset, learning_rate,
                                                 max_epoch=max_epoch)
    plot_mean_square_errors(epoch_num, mean_squared_errors, error_margin)

    evaluate_model(adaline, get_dataset(noise_data_number=2, unipolar=False), unipolar=False)


def plot_mean_square_errors(epoch_num, mean_squared_errors, error_margin):
    plt.plot(np.arange(1, epoch_num + 1), mean_squared_errors, '*--')
    plt.plot(np.arange(1, epoch_num + 1), [error_margin] * epoch_num, '--', lw=1.1, c='grey')
    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run_training()
