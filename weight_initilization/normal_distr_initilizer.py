from typing import List

import numpy as np

from weight_initilization.basa_initializer import Initializer


class NormalDistributionInitializer(Initializer):

    def __init__(self, sd: float = 1.0):
        super().__init__()
        self._sd = sd

    def init_weights(self, sizes: List[int]) -> List[np.ndarray]:
        return [np.random.normal(0, self._sd, size=(sizes[i + 1], sizes[i])) for i in range(len(sizes) - 1)]

    def init_biases(self, sizes: List[int]) -> List[np.ndarray]:
        return [np.random.normal(0, self._sd, size=(sizes[i + 1], 1)) for i in range(len(sizes) - 1)]
