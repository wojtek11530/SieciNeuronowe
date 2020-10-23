import numpy as np


def diff_loss_function(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_real - y_pred


def squared_loss_function(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.power(y_real - y_pred, 2)


def cross_entropy(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return -np.sum(y_real * np.log(y_pred), axis=1, keepdims=False)
