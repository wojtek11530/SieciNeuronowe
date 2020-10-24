from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from models.simple_models.base_simple_model import BaseSimpleModel


def evaluate_model(model: BaseSimpleModel, dataset: Tuple[np.ndarray, np.array], unipolar: bool = True):
    print('\nModel evaluation:')
    x_set, y_set = dataset
    y_pred = model(x_set)
    for x, y, y_hat in zip(x_set, y_set, y_pred):
        print(f"x: {x}, y: {y}, y_hat: {y_hat}")
    print()
    plot_plane(model, unipolar=unipolar)


def plot_plane(model: BaseSimpleModel, show: bool = True, title: str = '', unipolar: bool = True):
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

    plt.plot([], 'o', markersize=2, color='b', label=r'$\hat{y}=1.0$')
    plt.plot([], 'o', markersize=2, color='r', label=r'$\hat{{y}}={0}$'.format((zero_val)))
    plot_separating_lin(model, zero_val)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title)
    # plt.axis('equal')
    plt.xlim(zero_val - step, 1 + step)
    plt.ylim(zero_val - step, 1 + step)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    if show:
        plt.show()


def plot_separating_lin(model: BaseSimpleModel, zero_val: float):
    x1 = np.linspace(zero_val, 1, 100)
    w1 = model.weights[0]
    w2 = model.weights[1]
    b = model.bias
    plt.plot(x1, 1 / w2 * (-w1 * x1 - b), color='grey')
