from typing import List, Dict

import numpy as np

from optimizers.base_optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, parameters: Dict[str, List[np.ndarray]] = None, learning_rate: float = 1e-2):
        super().__init__(parameters)
        self._lr = learning_rate
        self._previous_parameters_squared_gradients_sum = {}
        self._eps = 10e-8
        for parameters_name, parameters_values in self._parameters.items():
            self._previous_parameters_squared_gradients_sum[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]

    def update_parameters(self, parameters_changes: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        for parameters_name, parameters_gradients in parameters_changes.items():
            self._previous_parameters_squared_gradients_sum[parameters_name] = \
                [squared_gradients_sum + np.square(gradient) for squared_gradients_sum, gradient in
                 zip(self._previous_parameters_squared_gradients_sum[parameters_name], parameters_gradients)]

        for parameters_name, parameters_gradients in parameters_changes.items():
            self._parameters[parameters_name] = \
                [params - self._lr / np.sqrt(squared_gradients_sum + self._eps) * gradient for
                 params, gradient, squared_gradients_sum
                 in zip(self._parameters[parameters_name], parameters_gradients,
                        self._previous_parameters_squared_gradients_sum[parameters_name])]

        return self._parameters

    def set_parameters(self, parameters: Dict = None):
        self._parameters = parameters
        for parameters_name, parameters_values in self._parameters.items():
            self._previous_parameters_squared_gradients_sum[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]
