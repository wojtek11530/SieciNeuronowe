import numpy as np


def unipolar_activation(z: np.ndarray) -> np.ndarray:
    return np.array(list(map(lambda x: 1 if x > 0 else 0, z)))


def bipolar_activation(z: np.ndarray) -> np.ndarray:
    return np.array(list(map(lambda x: 1 if x > 0 else -1, z)))
