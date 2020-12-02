import math
from typing import Tuple

import numpy as np


class MaxPool2D:
    def __init__(self, kernel_size: int, padding: int, stride: int):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.max_indexes = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out, max_indexes = map(list, zip(*[self.max_pool2d(x_channel) for x_channel in x]))
        self.max_indexes = np.array(max_indexes)
        return np.array(out)

    def __call__(self, x: np.ndarray):
        result = self.forward(x)
        return result

    def max_pool2d(self, x_channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        in_height, in_width = x_channel.shape
        padded_x = self._get_padded_x(in_height, in_width, x_channel)
        out_height = self._get_output_dim(in_height)
        out_width = self._get_output_dim(in_width)

        out = np.zeros(shape=(out_height, out_width))
        indexes = np.zeros(shape=(out_height, out_width, 2))

        i_out = 0
        i = 0
        while i <= padded_x.shape[0] - self.kernel_size:
            j = 0
            j_out = 0
            while j <= padded_x.shape[1] - self.kernel_size:
                window = padded_x[i: i + self.kernel_size, j: j + self.kernel_size]
                max_value = np.max(window)
                index = np.argwhere(window == max_value)
                out[i_out, j_out] = max_value
                indexes[i_out, j_out] = index

                j += self.stride
                j_out += 1

            i += self.stride
            i_out += 1

        return out, indexes

    def _get_padded_x(self, in_height: int, in_width: int, x: np.ndarray) -> np.ndarray:
        padded_x = np.zeros(shape=(in_height + 2 * self.padding, in_width + 2 * self.padding))
        padded_x[self.padding:self.padding + in_height, self.padding: self.padding + in_width] = x
        return padded_x

    def _get_output_dim(self, input_dim: int) -> int:
        return math.floor((input_dim - self.kernel_size + 2 * self.padding) / self.stride + 1)

    def __str__(self):
        return f'Max pooling - kernel: {self.kernel_size}, ' \
               f'padding: {self.padding}, stride: {self.stride}'
