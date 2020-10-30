from typing import Optional, Tuple

import numpy as np

from models.simple_models.base_simple_model import BaseSimpleModel


class Perceptron(BaseSimpleModel):

    def update_weight(self, x_set: np.ndarray, y_set: np.ndarray, lr: float) \
            -> Tuple[bool, Optional[float]]:
        any_weight_updated_in_epoch = False
        y_pred = self(x_set)
        deltas = self.loss_fn(y_set, y_pred)
        for x, delta in zip(x_set, deltas):
            weight_updated = self._update_weight_for_one_input(x, delta, lr)
            any_weight_updated_in_epoch = weight_updated or any_weight_updated_in_epoch

        return any_weight_updated_in_epoch, None

    def _update_weight_for_one_input(self, x: np.ndarray, delta: float, lr: float) -> bool:
        if delta != 0:
            self.weights += lr * delta * x
            self.bias += lr * delta
            return True
        else:
            return False

    def __str__(self):
        return 'weights: ' + ', '.join([str(w) for w in self.weights]) + ', bias: ' + str(self.bias)
