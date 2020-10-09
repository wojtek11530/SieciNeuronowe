from typing import Tuple

import numpy as np

from utils.random import get_random_float


def get_dataset(noise_data_number: int = 10, unipolar: bool = True) -> Tuple[np.ndarray, np.array]:
    if unipolar:
        zero_val = 0.
    else:
        zero_val = -1.

    x = [[1., 1.], [zero_val, zero_val], [zero_val, 1.], [1., zero_val]]
    y = [1, zero_val, zero_val, zero_val]

    limit = 0.05
    for i in range(noise_data_number):
        x.append([get_random_float(1 - limit, 1 + limit), get_random_float(1 - limit, 1 + limit)])
        y.append(1)
    for i in range(noise_data_number):
        x.append([get_random_float(zero_val - limit, zero_val + limit),
                  get_random_float(zero_val - limit, zero_val + limit)])
        y.append(zero_val)
    for i in range(noise_data_number):
        x.append([get_random_float(zero_val - limit, zero_val + limit), get_random_float(1 - limit, 1 + limit)])
        y.append(zero_val)
    for i in range(noise_data_number):
        x.append([get_random_float(1 - limit, 1 + limit), get_random_float(zero_val - limit, zero_val + limit)])
        y.append(zero_val)

    return np.array(x), np.array(y)
