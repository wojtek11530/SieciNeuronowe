from typing import List, Dict

import numpy as np

from optimizers.base_optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters: Dict[str, List[np.ndarray]] = None, learning_rate: float = 1e-2):
        super().__init__(parameters)
        self._lr = learning_rate

    def update_parameters(self, parameters_changes: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        for parameters_name, changes in parameters_changes.items():
            self.parameters[parameters_name] = \
                [params - self._lr * change for params, change in zip(self.parameters[parameters_name], changes)]

        return self.parameters
