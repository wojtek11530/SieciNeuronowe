from abc import abstractmethod
from typing import List, Dict

import numpy as np


class Optimizer:
    def __init__(self, parameters: Dict = None):
        if parameters is None:
            self._parameters = {}
        else:
            self._parameters = parameters

    @abstractmethod
    def update_parameters(self, parameters_changes: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        pass

    def set_parameters(self, parameters: Dict = None):
        self._parameters = parameters
