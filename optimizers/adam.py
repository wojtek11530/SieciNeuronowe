from typing import List, Dict

import numpy as np

from optimizers.base_optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, parameters: Dict[str, List[np.ndarray]] = None,
                 learning_rate: float = 1e-2, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 10e-8):
        super().__init__(parameters)
        self._lr = learning_rate
        self._squared_gradients = {}
        self._gradients = {}
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._time_step = 0
        for parameters_name, parameters_values in self._parameters.items():
            self._squared_gradients[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]
            self._gradients[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]

    def update_parameters(self, parameters_changes: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        self._time_step += 1

        for parameters_name in parameters_changes.keys():
            self._gradients[parameters_name] = \
                [self._beta1 * prev_gradient + (1 - self._beta1) * gradients
                 for prev_gradient, gradients in
                 zip(self._gradients[parameters_name], parameters_changes[parameters_name])]

            self._squared_gradients[parameters_name] = \
                [self._beta2 * prev_squared_gradient + (1 - self._beta2) * np.square(gradients)
                 for prev_squared_gradient, gradients in
                 zip(self._squared_gradients[parameters_name], parameters_changes[parameters_name])]

        for parameters_name in parameters_changes.keys():
            self._parameters[parameters_name] = \
                [params - self._lr * self._get_corrected_gradient(gradient)
                 / (np.sqrt(self._get_corrected_squared_gradient(squared_gradient) + self._eps))
                 for params, gradient, squared_gradient
                 in zip(self._parameters[parameters_name],
                        self._gradients[parameters_name],
                        self._squared_gradients[parameters_name])
                 ]

        return self._parameters

    def set_parameters(self, parameters: Dict = None):
        self._parameters = parameters
        for parameters_name, parameters_values in self._parameters.items():
            self._squared_gradients[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]
            self._gradients[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]

    def _get_corrected_squared_gradient(self, squared_gradient: np.ndarray) -> np.ndarray:
        return squared_gradient / (1 - np.power(self._beta2, self._time_step))

    def _get_corrected_gradient(self, gradient: np.ndarray) -> np.ndarray:
        return gradient / (1 - np.power(self._beta1, self._time_step))
