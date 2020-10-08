import numpy as np

from functions.activation_functions import unipolar_activation
from functions.loss_functions import loss_function
from utils.random import get_random_float


class Perceptron:
    def __init__(self, input_dim: int, weight_limit: float = 1, activation_fn=unipolar_activation):
        if weight_limit > 1:
            weight_limit = 1

        self.weights = get_random_float(min_val=-weight_limit, max_val=weight_limit, shape=input_dim)
        self.bias = get_random_float(min_val=-weight_limit, max_val=weight_limit)

        self.loss_fn = loss_function
        self.activation_fn = activation_fn

    def forward(self, x: np.ndarray) -> float:
        z = np.dot(x, self.weights) + self.bias
        return self.activation_fn(z)

    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: float) -> bool:
        any_weight_updated_in_epoch = False
        for x, y in zip(x_set, y_set):
            weight_updated = self._update_weight_for_one_input(x, y, lr)
            any_weight_updated_in_epoch = weight_updated or any_weight_updated_in_epoch

        return any_weight_updated_in_epoch

    def _update_weight_for_one_input(self, x: np.ndarray, y: float, lr: float) -> bool:
        y_pred = self(x)
        delta = self.loss_fn(y, y_pred)
        if delta != 0:
            self.weights += lr * delta * x
            self.bias += lr * delta
            return True
        else:
            return False

    def __call__(self, x: np.ndarray):
        result = self.forward(x)
        return result

    def __str__(self):
        return 'weights: ' + ', '.join([str(w) for w in self.weights]) + ', bias: ' + str(self.bias)
