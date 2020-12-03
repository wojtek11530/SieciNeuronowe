import pickle as pkl
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from functions.activation_functions import \
    (relu, relu_derivative, sigmoid, sigmoid_derivative, softmax, tanh, tanh_derivative)
from functions.loss_functions import cross_entropy, cross_entropy_loss_with_softmax_derivative
from models.neural_network_models.convolution_net_components.convolution_layer import Conv2D, convolve2d
from models.neural_network_models.convolution_net_components.pooling_layer import MaxPool2D
from models.neural_network_models.neural_network_base import NeuralNetworkBaseModel
from optimizers.base_optimizer import Optimizer
from optimizers.sgd import SGD
from weight_initilization.basa_initializer import Initializer
from weight_initilization.normal_distr_initilizer import NormalDistributionInitializer


class ConvolutionalNet(NeuralNetworkBaseModel):
    def __init__(self, input_dim: Tuple[int, int], fc_input_dim: int, output_dim: int, hidden_dims: List[int],
                 kernel_number: int,
                 init_parameters_sd: float = 1.0,
                 activation_functions: Optional[List[Callable]] = None,
                 optimizer: Optimizer = SGD(),
                 initializer: Optional[Initializer] = None):

        if initializer is None:
            initializer = NormalDistributionInitializer(init_parameters_sd)

        self.convolutional_layer = Conv2D(out_channels=kernel_number, kernel_size=3, padding=1, stride=1)
        self.convolutional_layer.init_parameters(input_dim)

        self.max_pooling = MaxPool2D(kernel_size=2, padding=0, stride=2)

        self.fc_input_dim = fc_input_dim
        sizes = [fc_input_dim] + hidden_dims + [output_dim]
        self.weights = initializer.init_weights(sizes)
        self.biases = initializer.init_biases(sizes)

        if activation_functions is None:
            self.activation_functions = [sigmoid] * (len(self.weights) - 1) + [softmax]
        else:
            self.activation_functions = activation_functions + [softmax]

        self.optimizer = optimizer
        self.optimizer.set_parameters({'weights': self.weights, 'biases': self.biases,
                                       'conv_weights': self.convolutional_layer.weights,
                                       'conv_biases': self.convolutional_layer.biases})

    def forward(self, x: np.ndarray) -> np.ndarray:

        out = self.convolutional_layer(x)
        out = relu(out)
        # out = self.max_pooling(out)
        a = out.reshape((-1, 1))
        for weights, bias, fn in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(weights, a) + bias
            a = fn(z)
        return a

    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: Optional[float] = None) \
            -> Tuple[bool, Optional[float]]:

        total_weights_change = [np.zeros(w.shape) for w in self.weights]
        total_biases_change = [np.zeros(b.shape) for b in self.biases]
        total_filters_weights_change = [np.zeros(w.shape) for w in self.convolutional_layer.weights]
        total_filters_biases_change = np.zeros(self.convolutional_layer.biases.shape)

        for x, y_real in zip(x_set, y_set):
            all_weight_changes, all_biases_changes, filters_weights_changes, filters_biases_changes = \
                self._backpropagate(x, y_real)
            total_weights_change = [cur_weight_change + weight_change for cur_weight_change, weight_change in
                                    zip(total_weights_change, all_weight_changes)]
            total_biases_change = [cur_biases_change + biases_change for cur_biases_change, biases_change in
                                   zip(total_biases_change, all_biases_changes)]
            total_filters_weights_change = [cur_weight_change + weight_change for cur_weight_change, weight_change in
                                            zip(total_filters_weights_change, filters_weights_changes)]
            total_filters_biases_change += filters_biases_changes

        batch_size = len(x_set)
        total_weights_change = [weight_change / batch_size for weight_change in total_weights_change]
        total_biases_change = [biases_change / batch_size for biases_change in total_biases_change]
        total_filters_weights_change = [weight_change / batch_size
                                        for weight_change in total_filters_weights_change]
        total_filters_biases_change /= batch_size

        parameters_changes = {'weights': total_weights_change, 'biases': total_biases_change,
                              'conv_weights': total_filters_weights_change,
                              'conv_biases': total_filters_biases_change}

        updated_parameters = self.optimizer.update_parameters(parameters_changes)
        self.weights = updated_parameters['weights']
        self.biases = updated_parameters['biases']
        self.convolutional_layer.weights = updated_parameters['conv_weights']
        self.convolutional_layer.biases = np.array(updated_parameters['conv_biases'])

        y_pred = np.array([self.forward(x) for x in x_set])
        losses_after_update = self.calculate_losses(y_pred, y_set)
        return True, float(np.mean(losses_after_update))

    def calculate_losses(self, y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
        return np.array([cross_entropy(y_true, y_hat)
                         for y_hat, y_true in zip(y_pred, y_real)])

    def _backpropagate(self, x: np.ndarray, y_real: np.ndarray) -> \
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        sample_weights_change = [np.zeros(w.shape) for w in self.weights]
        sample_biases_change = [np.zeros(b.shape) for b in self.biases]

        conv_out, activations, zs = self._get_conv_layer_output_activations_and_z_for_layers(x)

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

        deltas = np.dot(self.weights[0].transpose(), deltas) * relu_derivative(conv_out)
        deltas = deltas.reshape(self.convolutional_layer.get_output_dim(x))

        sample_filter_weights_changes = self.convolutional_layer.get_weights_gradients(x, deltas)
        sample_filter_biases_changes = np.array([np.sum(output_channels_deltas) for output_channels_deltas in deltas])

        return sample_weights_change, sample_biases_change, sample_filter_weights_changes, sample_filter_biases_changes

    def _get_conv_layer_output_activations_and_z_for_layers(self, x: np.ndarray) -> \
            Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:

        conv_out = self.convolutional_layer(x).reshape((-1, 1))
        conv_out_activated = relu(conv_out)
        # out = self.max_pooling(out)
        a = conv_out_activated
        activations = [a]
        zs = []

        for weights, bias, fn in zip(self.weights, self.biases, self.activation_functions):
            z = np.dot(weights, a) + bias
            a = fn(z)
            zs.append(a)
            activations.append(a)

        return conv_out, activations, zs

    def save_model(self, file_name: str) -> None:
        model_dict = self.get_model_parameters_as_dict()
        with open(file_name, 'wb') as handle:
            pkl.dump(model_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def get_model_parameters_as_dict(self) -> Dict:
        model_dict = {'conv_weights': self.convolutional_layer.weights, 'conv_biases': self.convolutional_layer.biases,
                      'weights': self.weights, 'biases': self.biases, 'act_fn': self.activation_functions}
        return model_dict

    def load_model(self, file_name: str) -> None:
        with open(file_name, 'rb') as handle:
            model_dict = pkl.load(handle)
            self.set_model_parameters_from_dict(model_dict)

    def set_model_parameters_from_dict(self, model_dict: Dict) -> None:
        self.convolutional_layer.weights = model_dict['conv_weights']
        self.convolutional_layer.biases = model_dict['conv_biases']
        self.weights = model_dict['weights']
        self.biases = model_dict['biases']
        self.activation_functions = model_dict['act_fn']

    def __str__(self):
        dims = []
        for weight in self.weights:
            dims.append(str(weight.shape[1]))
            dims.append(str(weight.shape[0]))

        dims = dims[:-2] + [dims[-1]]
        string = 'ConvNet model:' \
                 + '\n - ' + str(self.convolutional_layer) \
                 + '\n - FC_dims: ' + ', '.join(dims) \
                 + '\n - activation_functions: ' + ', '.join(str(fn) for fn in self.activation_functions)
        return string
