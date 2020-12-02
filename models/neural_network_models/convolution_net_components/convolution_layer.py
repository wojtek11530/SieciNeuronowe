import math
from typing import Tuple

import numpy as np

from functions.activation_functions import relu


class Conv2D:
    def __init__(self, out_channels: int, kernel_size: int, padding: int, stride: int):
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weights = [np.random.normal(0, 1, size=(kernel_size, kernel_size)) for i in range(out_channels)]
        self.biases = [np.random.normal(0, 1, size=(1, 1)) for i in range(out_channels)]
        self.activation_fn = relu

    def init_parameters(self, input_dim: Tuple[int, int]):
        input_n = input_dim[0] * input_dim[1]
        self.weights = [np.random.normal(0, np.sqrt(2 / input_n), size=(self.kernel_size, self.kernel_size))
                        for i in range(self.out_channels)]
        self.biases = [np.random.normal(0, np.sqrt(2 / input_n), size=(1, 1))
                       for i in range(self.out_channels)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.array([self.convolve2d(x, kernel) + bias for kernel, bias in zip(self.weights, self.biases)])
        out = self.activation_fn(out)
        return out

    def __call__(self, x: np.ndarray):
        result = self.forward(x)
        return result

    def convolve2d(self, x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        in_height, in_width = x.shape
        padded_x = self._get_padded_x(in_height, in_width, x)
        out_height = self._get_output_dim(in_height)
        out_width = self._get_output_dim(in_width)

        out = np.zeros(shape=(out_height, out_width))
        i_out = 0
        i = 0
        while i <= padded_x.shape[0] - self.kernel_size:
            j = 0
            j_out = 0
            while j <= padded_x.shape[1] - self.kernel_size:
                out[i_out, j_out] = np.sum(padded_x[i: i + self.kernel_size, j: j + self.kernel_size] * kernel)
                j += self.stride
                j_out += 1

            i += self.stride
            i_out += 1

        return out

    def _get_padded_x(self, in_height: int, in_width: int, x: np.ndarray) -> np.ndarray:
        padded_x = np.zeros(shape=(in_height + 2 * self.padding, in_width + 2 * self.padding))
        padded_x[self.padding:self.padding + in_height, self.padding: self.padding + in_width] = x
        return padded_x

    def _get_output_dim(self, input_dim: int) -> int:
        return math.floor((input_dim - self.kernel_size + 2 * self.padding) / self.stride + 1)

    def __str__(self):
        return f'Conv layer - out: {self.out_channels}, kernel: {self.kernel_size}, ' \
               f'padding: {self.padding}, stride: {self.stride}'