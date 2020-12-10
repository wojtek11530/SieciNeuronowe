import math
from typing import Tuple, List

import numpy as np
from scipy.signal import convolve2d as scipy_convolve2d


class Conv2D:
    def __init__(self, out_channels: int, kernel_size: int, padding: int, stride: int):
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weights = [np.random.normal(0, 1, size=(kernel_size, kernel_size)) for i in range(out_channels)]
        self.biases = np.random.normal(0, 1, size=self.out_channels)

    def init_parameters(self, input_dim: Tuple[int, int]):
        input_n = input_dim[0] * input_dim[1]
        self.weights = [np.random.normal(0, np.sqrt(2 / input_n), size=(self.kernel_size, self.kernel_size))
                        for i in range(self.out_channels)]
        self.biases = np.random.normal(0, np.sqrt(2 / input_n), size=self.out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.array([convolve2d(x, kernel, stride=self.stride, padding=self.padding) + bias for kernel, bias in
                        zip(self.weights, self.biases)])
        return out

    def __call__(self, x: np.ndarray):
        result = self.forward(x)
        return result

    def get_output_dim(self, x: np.ndarray) -> Tuple[int, int, int]:
        in_height, in_width = x.shape
        out_height = math.floor((in_height - self.kernel_size + 2 * self.padding) / self.stride + 1)
        out_width = math.floor((in_width - self.kernel_size + 2 * self.padding) / self.stride + 1)

        return self.out_channels, out_height, out_width

    def get_weights_gradients(self, x: np.ndarray, deltas: np.ndarray) -> List[np.ndarray]:
        updates = [convolve2d(np.rot90(x, 2), output_channels_deltas, stride=self.stride, padding=self.padding)
                   for output_channels_deltas in deltas]
        return updates

    def __str__(self):
        return f'Conv layer - out: {self.out_channels}, kernel: {self.kernel_size}, ' \
               f'padding: {self.padding}, stride: {self.stride}'


def convolve2d(x: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    padded_x = get_padded_x(x, padding=padding)

    out = scipy_convolve2d(padded_x, np.rot90(kernel, 2), mode='valid')

    # in_height, in_width = x.shape
    # kernel_height, kernel_width = kernel.shape
    # out_height = math.floor((in_height - kernel_height + 2 * padding) / stride + 1)
    # out_width = math.floor((in_width - kernel_width + 2 * padding) / stride + 1)

    # out = np.zeros(shape=(out_height, out_width))
    # i_out = 0
    # i = 0
    # while i <= padded_x.shape[0] - kernel_height:
    #     j = 0
    #     j_out = 0
    #     while j <= padded_x.shape[1] - kernel_width:
    #         out[i_out, j_out] = np.sum(padded_x[i: i + kernel_height, j: j + kernel_width] * kernel)
    #         j += stride
    #         j_out += 1
    #
    #     i += stride
    #     i_out += 1

    return out


def get_padded_x(x: np.ndarray, padding: int = 0) -> np.ndarray:
    in_height, in_width = x.shape
    padded_x = np.zeros(shape=(in_height + 2 * padding, in_width + 2 * padding))
    padded_x[padding:padding + in_height, padding:padding + in_width] = x
    return padded_x


def get_convolve_output_dim(input_dim: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    return math.floor((input_dim - kernel_size + 2 * padding) / stride + 1)
