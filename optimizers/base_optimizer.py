from abc import abstractmethod
from typing import List, Dict

import numpy as np


class Optimizer:
    def __init__(self, parameters: Dict = None):
        self.parameters = parameters

    @abstractmethod
    def update_parameters(self, parameters_changes: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        pass
