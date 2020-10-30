from typing import List, Optional, Tuple

import numpy as np

from models.simple_models.base_simple_model import BaseSimpleModel


def train_model(model: BaseSimpleModel, dataset: Tuple[np.ndarray, np.array], lr: float,
                max_epoch: Optional[int] = None, verbose: bool = True) \
        -> Tuple[int, List[Optional[float]]]:
    continue_training = True
    epoch_num = 1
    errors = []
    print(f'Training started, lr={lr}, {type(model)}')
    while continue_training:
        if verbose:
            print(f'Epoch {epoch_num}', end='')
            print(f',\t {str(model)}')

        continue_training, error = train_one_epoch(model, dataset, lr)
        if verbose and error:
            print(f'Error: {str(error)}')
        errors.append(error)

        if continue_training:
            epoch_num += 1

        if max_epoch and epoch_num > max_epoch:
            epoch_num = epoch_num - 1
            continue_training = False

    print(f'Training ends. Epochs total={epoch_num}')
    return epoch_num, errors


def train_one_epoch(model: BaseSimpleModel, dataset: Tuple[np.ndarray, np.ndarray], lr: float) \
        -> Tuple[bool, Optional[float]]:
    x_set, y_set = dataset
    continue_training, error = model.update_weight(x_set, y_set, lr)
    return continue_training, error
