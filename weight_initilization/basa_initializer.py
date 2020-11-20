from abc import abstractmethod
from typing import List

import numpy as np


class Initializer:

    @abstractmethod
    def init_weights(self, sizes: List[int]) -> List[np.ndarray]:
        pass

    @abstractmethod
    def init_biases(self, sizes: List[int]) -> List[np.ndarray]:
        pass
