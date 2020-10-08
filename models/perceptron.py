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

    def __call__(self, x: np.ndarray):
        result = self.forward(x)
        return result

    def update_weight(self, x: np.ndarray, y: float, lr: float) -> bool:
        y_pred = self(x)
        delta = self.loss_fn(y, y_pred)
        if delta != 0:
            self.weights += lr * delta * x
            self.bias += lr * delta
            return True
        else:
            return False

    def __str__(self):
        return 'weights: ' + ', '.join([str(w) for w in self.weights]) + ', bias: ' + str(self.bias)
