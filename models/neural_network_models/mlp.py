import pickle as pkl
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from functions.activation_functions import relu_derivative, sigmoid, sigmoid_derivative, softmax, tanh, tanh_derivative
from functions.loss_functions import cross_entropy, cross_entropy_loss_with_softmax_derivative
from models.neural_network_models.neural_network_base import NeuralNetworkBaseModel
from optimizers.base_optimizer import Optimizer
from optimizers.nestorov_momentum import NestorovMomentum
from optimizers.sgd import SGD
from weight_initilization.basa_initializer import Initializer
from weight_initilization.normal_distr_initilizer import NormalDistributionInitializer


class MLP(NeuralNetworkBaseModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 init_parameters_sd: float = 1.0,
                 activation_functions: Optional[List[Callable]] = None,
                 optimizer: Optimizer = SGD(),
                 initializer: Optional[Initializer] = None):

        self.input_dim = input_dim
        sizes = [input_dim] + hidden_dims + [output_dim]

        if initializer is None:
            initializer = NormalDistributionInitializer(init_parameters_sd)

        self.weights = initializer.init_weights(sizes)
        self.biases = initializer.init_biases(sizes)

        if activation_functions is None:
            self.activation_functions = [sigmoid] * (len(self.weights) - 1) + [softmax]
        else:
            self.activation_functions = activation_functions + [softmax]

        self.optimizer = optimizer
        self.optimizer.set_parameters({'weights': self.weights, 'biases': self.biases})

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x.reshape((-1, 1))
        for weights, bias, fn in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(weights, a) + bias
            a = fn(z)
        return a

    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: Optional[float] = None) \
            -> Tuple[bool, Optional[float]]:

        total_weights_change = [np.zeros(w.shape) for w in self.weights]
        total_biases_change = [np.zeros(b.shape) for b in self.biases]

        for x, y_real in zip(x_set, y_set):
            sample_change_weights, sample_change_biases = self._backpropagate(x, y_real)
            total_weights_change = [cur_weight_change + weight_change for cur_weight_change, weight_change in
                                    zip(total_weights_change, sample_change_weights)]
            total_biases_change = [cur_biases_change + biases_change for cur_biases_change, biases_change in
                                   zip(total_biases_change, sample_change_biases)]

        batch_size = len(x_set)
        total_weights_change = [weight_change / batch_size for weight_change in total_weights_change]
        total_biases_change = [biases_change / batch_size for biases_change in total_biases_change]

        parameters_changes = {'weights': total_weights_change, 'biases': total_biases_change}
        updated_parameters = self.optimizer.update_parameters(parameters_changes)
        self.weights = updated_parameters['weights']
        self.biases = updated_parameters['biases']

        y_pred = np.array([self.forward(x) for x in x_set])
        losses_after_update = self.calculate_losses(y_pred, y_set)
        return True, float(np.mean(losses_after_update))

    def calculate_losses(self, y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
        return np.array([cross_entropy(y_true, y_hat)
                         for y_hat, y_true in zip(y_pred, y_real)])

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

            if isinstance(self.optimizer, NestorovMomentum):
                momentum_rate = self.optimizer.momentum_rate
                prev_weights_changes = self.optimizer.previous_parameters_updates['weights'][layer_index + 1]
                next_layer_weights = self.weights[layer_index + 1] - momentum_rate * prev_weights_changes
            else:
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

        if isinstance(self.optimizer, NestorovMomentum):
            momentum_rate = self.optimizer.momentum_rate
            prev_weights_changes = self.optimizer.previous_parameters_updates['weights']
            prev_biases = self.optimizer.previous_parameters_updates['biases']
            for weights, weight_changes, bias, bias_changes, fn in \
                    zip(self.weights, prev_weights_changes, self.biases, prev_biases, self.activation_functions):
                used_weights = weights - momentum_rate * weight_changes
                used_biases = bias - momentum_rate * bias_changes
                z = np.dot(used_weights, a) + used_biases
                a = fn(z)
                zs.append(a)
                activations.append(a)
        else:
            for weights, bias, fn in zip(self.weights, self.biases, self.activation_functions):
                z = np.dot(weights, a) + bias
                a = fn(z)
                zs.append(a)
                activations.append(a)

        return activations, zs

    def save_model(self, file_name: str) -> None:
        model_dict = self.get_model_parameters_as_dict()
        with open(file_name, 'wb') as handle:
            pkl.dump(model_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def get_model_parameters_as_dict(self) -> Dict:
        model_dict = {'weights': self.weights, 'biases': self.biases, 'act_fn': self.activation_functions}
        return model_dict

    def load_model(self, file_name: str) -> None:
        with open(file_name, 'rb') as handle:
            model_dict = pkl.load(handle)
            self.set_model_parameters_from_dict(model_dict)

    def set_model_parameters_from_dict(self, model_dict: Dict) -> None:
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
