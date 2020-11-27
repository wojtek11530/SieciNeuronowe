from typing import List, Dict

import numpy as np

from optimizers.base_optimizer import Optimizer


class Adadelta(Optimizer):
    def __init__(self, parameters: Dict[str, List[np.ndarray]] = None, gamma: float = 0.9, eps: float = 1e-5):
        super().__init__(parameters)
        self._parameters_squared_gradients_mean = {}
        self._parameters_squared_updates_mean = {}
        self._gamma = gamma
        self._eps = eps
        for parameters_name, parameters_values in self._parameters.items():
            self._parameters_squared_gradients_mean[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]
            self._parameters_squared_updates_mean[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]

    def update_parameters(self, parameters_changes: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        for parameters_name in parameters_changes.keys():
            self._parameters_squared_gradients_mean[parameters_name] = \
                [self._calculate_decay_average(squared_gradients_mean, gradients)
                 for squared_gradients_mean, gradients in
                 zip(self._parameters_squared_gradients_mean[parameters_name], parameters_changes[parameters_name])]

        for parameters_name in parameters_changes.keys():
            parameters_changes[parameters_name] = \
                [- self._RMS(sq_param_updates_mean) / self._RMS(sq_grad_mean) * gradient
                 for sq_grad_mean, sq_param_updates_mean, gradient in
                 zip(self._parameters_squared_gradients_mean[parameters_name],
                     self._parameters_squared_updates_mean[parameters_name],
                     parameters_changes[parameters_name])]

        for parameters_name in parameters_changes.keys():
            self._parameters[parameters_name] = \
                [params + param_change for params, param_change
                 in zip(self._parameters[parameters_name], parameters_changes[parameters_name])]

            self._parameters_squared_updates_mean[parameters_name] = \
                [self._calculate_decay_average(squared_updates_mean, update) for squared_updates_mean, update in
                 zip(self._parameters_squared_updates_mean[parameters_name], parameters_changes[parameters_name])]

        return self._parameters

    def set_parameters(self, parameters: Dict = None):
        self._parameters = parameters
        for parameters_name, parameters_values in self._parameters.items():
            self._parameters_squared_gradients_mean[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]
            self._parameters_squared_updates_mean[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]

    def _RMS(self, current_mean: np.ndarray) -> np.ndarray:
        return np.sqrt(current_mean + self._eps)

    def _calculate_decay_average(self, current_average: np.ndarray, new_element: np.ndarray) -> np.ndarray:
        return self._gamma * current_average + (1 - self._gamma) * np.square(new_element)
