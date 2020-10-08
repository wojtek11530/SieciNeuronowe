from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from models.perceptron import Perceptron


def evaluate_model(perceptron: Perceptron, dataset: Tuple[np.ndarray, np.array]):
    print('\nModel evaluation:')
    x_set, y_set = dataset
    for x, y in zip(x_set, y_set):
        y_pred = perceptron(x)
        print(f"x: {x}, y: {y}, y_pred: {y_pred}")
    print()
    plot_plane(perceptron)


def plot_plane(perceptron: Perceptron, show: bool = True, title: str = ''):
    step = 0.05
    for x1 in np.arange(0, 1 + step, step):
        for x2 in np.arange(0, 1 + step, step):
            y = perceptron(np.array([x1, x2]))
            color = 'b' if y == 1 else 'r'
            plt.plot(x1, x2, 'o', markersize=2, color=color)

    x1 = np.linspace(0, 1, 100)
    w1 = perceptron.weights[0]
    w2 = perceptron.weights[1]
    b = perceptron.bias
    plt.plot(x1, 1 / w2 * (-w1 * x1 - b), color='grey')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title)
    plt.axis('equal')
    plt.xlim(0 - step, 1 + step)
    plt.ylim(0 - step, 1 + step)
    if show:
        plt.show()
