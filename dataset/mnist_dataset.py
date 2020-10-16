from typing import Tuple

import numpy as np

from mnist import MNIST


def get_train_dataset() -> Tuple[np.ndarray, np.array]:
    mndata = MNIST('.')
    mndata.gz = True
    images, labels = mndata.load_training()
    return np.array(images), np.array(labels)
