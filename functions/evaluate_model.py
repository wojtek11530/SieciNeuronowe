from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from models.adaline import Adaline
from models.base import BaseModel
from models.perceptron import Perceptron


def evaluate_model(model: BaseModel, dataset: Tuple[np.ndarray, np.array], unipolar: bool = True):
    print('\nModel evaluation:')
    x_set, y_set = dataset
    y_pred = model(x_set)
    for x, y, y_hat in zip(x_set, y_set, y_pred):
        print(f"x: {x}, y: {y}, y_hat: {y_hat}")
    print()
    plot_plane(model, unipolar=unipolar)


def plot_plane(model: BaseModel, show: bool = True, title: str = '', unipolar: bool = True):
    step = 0.05
    if unipolar:
        zero_val = 0.
    else:
        zero_val = -1.

    for x1 in np.arange(zero_val, 1 + step, step):
        for x2 in np.arange(zero_val, 1 + step, step):
            y = model(np.array([[x1, x2]]))
            color = 'b' if y == 1 else 'r'
            plt.plot(x1, x2, 'o', markersize=2, color=color)

    x1 = np.linspace(zero_val, 1, 100)
    if type(model) in [Perceptron, Adaline]:
        plot_separating_lin(model, x1)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title)
    plt.axis('equal')
    plt.xlim(zero_val - step, 1 + step)
    plt.ylim(zero_val - step, 1 + step)
    if show:
        plt.show()


def plot_separating_lin(model: Union[Perceptron, Adaline], x1: np.ndarray):
    w1 = model.weights[0]
    w2 = model.weights[1]
    b = model.bias
    plt.plot(x1, 1 / w2 * (-w1 * x1 - b), color='grey')
