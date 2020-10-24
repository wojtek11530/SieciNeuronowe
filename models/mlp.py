import pickle as pkl

from typing import Optional, Tuple, List, Union

import numpy as np

from functions.activation_functions import sigmoid, softmax, tanh, relu, sigmoid_derivative, relu_derivative, \
    tanh_derivative
from functions.loss_functions import cross_entropy_loss_with_softmax_derivative, cross_entropy
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x.reshape((-1, 1))
        for weights, bias, fn in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(weights, a) + bias
            a = fn(z)
        return a

    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: float) -> Tuple[bool, Optional[float]]:

        total_weights_change = [np.zeros(w.shape) for w in self.weights]
        total_biases_change = [np.zeros(b.shape) for b in self.biases]

        for x, y_real in zip(x_set, y_set):
            sample_change_weights, sample_change_biases = self._backpropagate(x, y_real)
            total_weights_change = [cur_weight_change + weight_change for cur_weight_change, weight_change in
                                    zip(total_weights_change, sample_change_weights)]
            total_biases_change = [cur_biases_change + biases_change for cur_biases_change, biases_change in
                                   zip(total_biases_change, sample_change_biases)]

        batch_size = len(x_set)
        self.weights = [weight - lr / batch_size * weights_change
                        for weight, weights_change in zip(self.weights, total_weights_change)]
        self.biases = [biases - lr / batch_size * biases_change
                       for biases, biases_change in zip(self.biases, total_biases_change)]

        losses_after_update = self._calculate_losses(x_set, y_set)
        return True, float(np.mean(losses_after_update))

    def _calculate_losses(self, x_set, y_set):
        return np.array([cross_entropy(y_real, self.forward(x))
                         for x, y_real in zip(x_set, y_set)])

    def _backpropagate(self, x: np.ndarray, y_real: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        sample_weights_change = [np.zeros(w.shape) for w in self.weights]
        sample_biases_change = [np.zeros(b.shape) for b in self.biases]

        activations, zs = self._get_activations_and_z_for_layers(x)

        deltas = cross_entropy_loss_with_softmax_derivative(activations[-1], y_real)
        sample_biases_change[-1] = deltas
        sample_weights_change[-1] = np.dot(deltas, activations[-2].transpose())

        layers_num = len(self.biases)
        for layer_index in reversed(range(layers_num - 1)):
            fn = self.activation_functions[layer_index]
            if fn == sigmoid:
                fn_derivative = sigmoid_derivative
            elif fn == tanh:
                fn_derivative = tanh_derivative
            else:
                fn_derivative = relu_derivative

            next_layer_weights = self.weights[layer_index + 1]
            layer_z = zs[layer_index]
            deltas = np.dot(next_layer_weights.transpose(), deltas) * fn_derivative(layer_z)
            sample_biases_change[layer_index] = deltas
            sample_weights_change[layer_index] = np.dot(deltas, activations[layer_index].transpose())

        return sample_weights_change, sample_biases_change

    def _get_activations_and_z_for_layers(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        a = x
        activations = [a]
        zs = []
        for weights, bias, fn in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(weights, a) + bias
            a = fn(z)
            zs.append(a)
            activations.append(a)
        return activations, zs

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
