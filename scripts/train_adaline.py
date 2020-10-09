import matplotlib.pyplot as plt
import numpy as np

from dataset.and_dataset import get_dataset
from functions.evaluate_model import evaluate_model
from functions.tran_model import train_model
from models.adaline import Adaline


def run_training():
    weight_limit = 1.0
    learning_rate = 0.1
    error_margin = 0.1
    max_epoch = 50
    dataset = get_dataset(noise_data_number=20, unipolar=False)

    adaline = Adaline(2, weight_limit=weight_limit)
    epoch_num, mean_squared_errors = train_model(adaline, dataset, learning_rate,
                                                 error_margin=error_margin,
                                                 max_epoch=max_epoch,
                                                 unipolar=False, plot_epoch=False)
    plot_mean_square_errors(epoch_num, mean_squared_errors)

    evaluate_model(adaline, get_dataset(noise_data_number=2, unipolar=False), unipolar=False)


def plot_mean_square_errors(epoch_num, mean_squared_errors):
    plt.plot(np.arange(1, epoch_num + 1), mean_squared_errors, '.-')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run_training()
