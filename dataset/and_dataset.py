from typing import Tuple

import numpy as np

from utils.random import get_random_float


def get_dataset(case_gen_data_number: int = 10) -> Tuple[np.ndarray, np.array]:
    x = [[1., 1.], [0., 0.], [0., 1.], [1., 0.]]
    y = [1, 0, 0, 0]

    limit = 0.05
    for i in range(case_gen_data_number):
        x.append([get_random_float(1 - limit, 1), get_random_float(1 - limit, 1)])
        y.append(1)
    for i in range(case_gen_data_number):
        x.append([get_random_float(0, limit), get_random_float(0, limit)])
        y.append(0)
    for i in range(case_gen_data_number):
        x.append([get_random_float(0, limit), get_random_float(1 - limit, 1)])
        y.append(0)
    for i in range(case_gen_data_number):
        x.append([get_random_float(1 - limit, 1), get_random_float(0, limit)])
        y.append(0)

    return np.array(x), np.array(y)
