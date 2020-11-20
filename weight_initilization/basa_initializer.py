from abc import abstractmethod
from typing import List

import numpy as np


class Initializer:

    @staticmethod
    @abstractmethod
    def init_weights(sizes: List[int]):
        pass

    @staticmethod
    @abstractmethod
    def init_biases(sizes: List[int]) -> List[np.ndarray]:
        pass
