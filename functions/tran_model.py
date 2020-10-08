from typing import Tuple

import numpy as np

from models.perceptron import Perceptron


def train_model(perceptron: Perceptron, dataset: Tuple[np.ndarray, np.array], lr: float, verbose: bool = True) -> int:
    any_weight_updated = True
    epoch_num = 1
    print(f'Training started, lr={lr}')
    while any_weight_updated:
        if verbose:
            print(f'Epoch {epoch_num}', end='')
            print(f',\t {str(perceptron)}')

        any_weight_updated = train_one_epoch(perceptron, dataset, lr)
        if any_weight_updated:
            epoch_num += 1

    print(f'No weights updated. Training ends. Epochs total={epoch_num}')
    return epoch_num


def train_one_epoch(perceptron: Perceptron, dataset: Tuple[np.ndarray, np.array], lr: float) -> bool:
    any_weight_updated_in_epoch = False
    x_set, y_set = dataset
    for x, y in zip(x_set, y_set):
        weight_updated = perceptron.update_weight(x, y, lr)
        any_weight_updated_in_epoch = weight_updated or any_weight_updated_in_epoch

    return any_weight_updated_in_epoch
