from typing import Tuple

import numpy as np

from functions.activation_functions import bipolar_activation
from models.simple_models.base_simple_model import BaseSimpleModel


class Adaline(BaseSimpleModel):
    def __init__(self, input_dim: int, weight_limit: float = 1, error_margin: float = 0.4):
        super().__init__(input_dim, weight_limit, bipolar_activation)
        self.error_margin = error_margin

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        return self.activation_fn(z)

    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: float) \
            -> Tuple[bool, float]:
        z = np.dot(x_set, self.weights) + self.bias
        deltas = self.loss_fn(y_set, z)
        mean_squared_error = float(np.mean(np.power(deltas, 2)))
        if mean_squared_error > self.error_margin:
            for x, delta in zip(x_set, deltas):
                self._update_weight_for_one_input(x, delta, lr)
            return True, mean_squared_error
        else:
            return False, mean_squared_error

    def _update_weight_for_one_input(self, x: np.ndarray, delta: float, lr: float) -> None:
        self.weights += 2 * lr * delta * x
        self.bias += 2 * lr * delta

    def __str__(self):
        return 'weights: ' + ', '.join([str(w) for w in self.weights]) + ', bias: ' + str(self.bias)
