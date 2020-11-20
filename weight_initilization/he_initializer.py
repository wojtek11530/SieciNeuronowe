from typing import List

import numpy as np

from weight_initilization.basa_initializer import Initializer


class HeInitializer(Initializer):

    def init_weights(self, sizes: List[int]) -> List[np.ndarray]:
        return [np.random.normal(0, np.sqrt(2 / sizes[i]), size=(sizes[i + 1], sizes[i]))
                for i in range(len(sizes) - 1)]

    def init_biases(self, sizes: List[int]) -> List[np.ndarray]:
        return [np.random.normal(0, np.sqrt(2 / sizes[i]), size=(sizes[i + 1], 1))
                for i in range(len(sizes) - 1)]
