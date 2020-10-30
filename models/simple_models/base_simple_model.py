from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np

from functions.activation_functions import unipolar_activation
from functions.loss_functions import diff_loss_function
from models.base import BaseModel
from utils.random import get_random_float


class BaseSimpleModel(BaseModel):
    def __init__(self, input_dim: int, weight_limit: float = 1, activation_fn=unipolar_activation):
        if weight_limit > 1:
            weight_limit = 1

        self.weights = get_random_float(min_val=-weight_limit, max_val=weight_limit, shape=input_dim)
        self.bias = get_random_float(min_val=-weight_limit, max_val=weight_limit)

        self.loss_fn = diff_loss_function
        self.activation_fn = activation_fn

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        return self.activation_fn(z)

    @abstractmethod
    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: float) \
            -> Tuple[bool, Optional[float]]:
        pass
