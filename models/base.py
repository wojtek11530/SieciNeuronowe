from abc import abstractmethod

import numpy as np


class BaseModel:
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: float, error_margin: float) -> bool:
        pass

    def __call__(self, x: np.ndarray):
        result = self.forward(x)
        return result
