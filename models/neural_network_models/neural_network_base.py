from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
from mypy.dmypy_server import Dict

from models.base import BaseModel


class NeuralNetworkBaseModel(BaseModel):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: Optional[float] = None) \
            -> Tuple[bool, Optional[float]]:
        pass

    @abstractmethod
    def calculate_losses(self, y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_model_parameters_as_dict(self) -> Dict:
        pass

    @abstractmethod
    def set_model_parameters_from_dict(self, model_dict: Dict) -> None:
        pass

    @abstractmethod
    def save_model(self, file_name: str) -> None:
        pass

    @abstractmethod
    def load_model(self, file_name: str) -> None:
        pass
