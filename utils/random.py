import numpy as np


def get_random_float(min_val: float, max_val: float, shape=None):
    return (max_val - min_val) * np.random.random(size=shape) + min_val
