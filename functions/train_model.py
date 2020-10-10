from typing import List, Optional, Tuple

import numpy as np

from functions.evaluate_model import plot_plane
from models.base import BaseModel


def train_model(model: BaseModel, dataset: Tuple[np.ndarray, np.array], lr: float, error_margin: float = 0,
                max_epoch: Optional[int] = None, verbose: bool = True, unipolar: bool = True,
                plot_epoch: bool = False) \
        -> Tuple[int, List[Optional[float]]]:
    continue_training = True
    epoch_num = 1
    errors = []
    print(f'Training started, lr={lr}, {type(model)}')
    while continue_training:
        if verbose:
            print(f'Epoch {epoch_num}', end='')
            print(f',\t {str(model)}')

        if epoch_num > 100:
            a = 2
            pass
        continue_training, error = train_one_epoch(model, dataset, lr, error_margin)
        if verbose and error:
            print(f'Error: {str(error)}')
        errors.append(error)

        if plot_epoch:
            plot_plane(model, unipolar=unipolar, title='Epoch ' + str(epoch_num))

        if continue_training:
            epoch_num += 1

        if max_epoch and epoch_num > max_epoch:
            epoch_num = epoch_num - 1
            continue_training = False

    print(f'Training ends. Epochs total={epoch_num}')
    return epoch_num, errors


def train_one_epoch(model: BaseModel, dataset: Tuple[np.ndarray, np.ndarray], lr: float, error_margin: float) \
        -> Tuple[bool, Optional[float]]:
    x_set, y_set = dataset
    continue_training, error = model.update_weight(x_set, y_set, lr, error_margin)
    return continue_training, error
