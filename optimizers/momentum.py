from typing import List, Dict

import numpy as np

from optimizers.base_optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, parameters: Dict[str, List[np.ndarray]] = None, learning_rate: float = 1e-2,
                 momentum_rate: float = 0.7):
        super().__init__(parameters)
        self._lr = learning_rate
        self._momentum_rate = momentum_rate
        self._previous_parameters_updates = {}
        for parameters_name, parameters_values in self._parameters.items():
            self._previous_parameters_updates[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]

    def update_parameters(self, parameters_changes: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        for parameters_name, changes in parameters_changes.items():
            parameters_changes[parameters_name] = \
                [self._momentum_rate * prev_change + self._lr * change for prev_change, change in
                 zip(self._previous_parameters_updates[parameters_name], changes)]

        for parameters_name, changes in parameters_changes.items():
            self._parameters[parameters_name] = \
                [params - change for params, change in zip(self._parameters[parameters_name], changes)]

        self._previous_parameters_updates = parameters_changes
        return self._parameters

    def set_parameters(self, parameters: Dict = None):
        self._parameters = parameters
        for parameters_name, parameters_values in self._parameters.items():
            self._previous_parameters_updates[parameters_name] = \
                [np.zeros_like(values) for values in parameters_values]
