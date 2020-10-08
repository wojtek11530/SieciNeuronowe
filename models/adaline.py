import numpy as np

from functions.activation_functions import bipolar_activation
from functions.loss_functions import loss_function
from models.base import BaseModel
from utils.random import get_random_float


class Adaline(BaseModel):
    def __init__(self, input_dim: int, weight_limit: float = 1):
        if weight_limit > 1:
            weight_limit = 1

        self.weights = get_random_float(min_val=-weight_limit, max_val=weight_limit, shape=input_dim)
        self.bias = get_random_float(min_val=-weight_limit, max_val=weight_limit)

        self.loss_fn = loss_function
        self.activation_fn = bipolar_activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        return self.activation_fn(z)

    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: float, error_margin: float) -> bool:
        y_pred = self(x_set)
        deltas = self.loss_fn(y_set, y_pred)
        mean_squared_error = np.mean(np.power(deltas, 2))
        if mean_squared_error > error_margin:
            for x, delta in zip(x_set, deltas):
                self._update_weight_for_one_input(x, delta, lr)
            return True
        else:
            return False

    def _update_weight_for_one_input(self, x: np.ndarray, delta: float, lr: float) -> None:
        self.weights += 2 * lr * delta * x
        self.bias += 2 * lr * delta

    def __str__(self):
        return 'weights: ' + ', '.join([str(w) for w in self.weights]) + ', bias: ' + str(self.bias)
