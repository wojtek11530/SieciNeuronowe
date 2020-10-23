import pickle as pkl

from typing import Optional, Tuple, List, Callable, Union

import numpy as np

from functions.activation_functions import sigmoid, softmax, tanh, relu
from models.base import BaseModel


class MLP(BaseModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 init_parameters_sd: float = 1.0,

                 activation_functions: Optional[List[Union[sigmoid, softmax, tanh, relu]]] = None):
        self.input_dim = input_dim
        sizes = [input_dim] + hidden_dims + [output_dim]

        self.weights = [np.random.normal(0, init_parameters_sd, size=(sizes[i + 1], sizes[i]))
                        for i in range(len(sizes) - 1)]
        self.biases = [np.random.normal(0, init_parameters_sd, size=(sizes[i + 1], 1)) for i in range(len(sizes) - 1)]

        if activation_functions is None:
            self.activation_functions = [sigmoid] * (len(self.weights) - 1) + [softmax]
        else:
            self.activation_functions = activation_functions + [softmax]

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        a = x.reshape((-1, 1))
        z = None
        for weights, bias, fn in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(weights, a) + bias
            a = fn(z)
        return a, z

    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: float, error_margin: float) \
            -> Tuple[bool, Optional[float]]:
        pass
        # for x, y_real in zip(x_set, y_set):
        #     z, a = self.forward(x)
        # deltas = self.loss_fn(y_set, z)

    def save_model(self, file_name: str) -> None:
        model_dict = {'weights': self.weights, 'biases': self.biases, 'act_fn': self.activation_functions}
        with open(file_name, 'wb') as handle:
            pkl.dump(model_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def load_model(self, file_name: str) -> None:
        with open(file_name, 'rb') as handle:
            model_dict = pkl.load(handle)
            self.weights = model_dict['weights']
            self.biases = model_dict['biases']
            self.activation_functions = model_dict['act_fn']

    def __str__(self):
        dims = []
        for weight in self.weights:
            dims.append(str(weight.shape[1]))
            dims.append(str(weight.shape[0]))

        dims = dims[:-2] + [dims[-1]]
        string = 'MLP model:\n - dims: ' + ', '.join(dims) + '\n - activation_functions: ' \
                 + ', '.join(str(fn) for fn in self.activation_functions)
        return string