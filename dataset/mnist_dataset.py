from typing import Tuple

import numpy as np

from mnist import MNIST

_MNIST_DIRECTORY = '.'


def get_train_dataset() -> Tuple[np.ndarray, np.array]:
    mndata = MNIST(_MNIST_DIRECTORY)
    mndata.gz = True
    images, labels = mndata.load_training()
    return np.array(images), np.array(labels)


def get_test_dataset() -> Tuple[np.ndarray, np.array]:
    mndata = MNIST(_MNIST_DIRECTORY)
    mndata.gz = True
    images, labels = mndata.load_testing()
    return np.array(images), np.array(labels)
