import numpy as np


def loss_function(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_real - y_pred


def squared_loss_function(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.power(y_real - y_pred, 2)
