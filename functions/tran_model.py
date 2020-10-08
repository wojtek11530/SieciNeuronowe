from typing import Tuple

import numpy as np

from models.base import BaseModel


def train_model(model: BaseModel, dataset: Tuple[np.ndarray, np.array], lr: float, error_margin: float = 0,
                verbose: bool = True) -> int:
    continue_training = True
    epoch_num = 1
    print(f'Training started, lr={lr}, {type(model)}')
    while continue_training:
        if verbose:
            print(f'Epoch {epoch_num}', end='')
            print(f',\t {str(model)}')

        continue_training = train_one_epoch(model, dataset, lr, error_margin)
        if continue_training:
            epoch_num += 1

    print(f'Training ends. Epochs total={epoch_num}')
    return epoch_num


def train_one_epoch(model: BaseModel, dataset: Tuple[np.ndarray, np.ndarray], lr: float, error_margin: float) -> bool:
    x_set, y_set = dataset
    continue_training = model.update_weight(x_set, y_set, lr, error_margin)
    return continue_training
