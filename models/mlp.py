from typing import Optional, Tuple, List, Callable

import numpy as np

from functions.activation_functions import sigmoid
from models.base import BaseModel


class MLP(BaseModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 init_parameters_sd: float = 1.0,
                 activation_functions: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None):
        sizes = [input_dim] + hidden_dims + [output_dim]

        self.weights = [np.random.normal(0, init_parameters_sd, size=(sizes[i + 1], sizes[i]))
                        for i in range(len(sizes) - 1)]
        self.biases = [np.random.normal(0, init_parameters_sd, size=(sizes[i + 1], 1)) for i in range(len(sizes) - 1)]

        if activation_functions is None:
            self.activation_functions = [sigmoid] * len(self.weights)
        else:
            self.activation_functions = activation_functions

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x.reshape((-1, 1))
        for weights, bias, fn in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(weights, a) + bias
            a = fn(z)

        return a

    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: float, error_margin: float) \
            -> Tuple[bool, Optional[float]]:
        pass

    def __str__(self):
        return 'MLP model'
