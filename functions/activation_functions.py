import numpy as np


def unipolar_activation(z: np.ndarray) -> np.ndarray:
    return np.array(list(map(lambda x: 1 if x > 0 else 0, z)))


def bipolar_activation(z: np.ndarray) -> np.ndarray:
    return np.array(list(map(lambda x: 1 if x > 0 else -1, z)))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / np.sum(np.exp(z))


def tanh(z: np.ndarray) -> np.ndarray:
    return 2.0 / (1.0 + np.exp(-2 * z)) - 1


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)
