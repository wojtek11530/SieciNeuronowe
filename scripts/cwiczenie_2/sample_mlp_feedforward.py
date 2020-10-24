import numpy as np

from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid, softmax
from models.mlp import MLP

x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

mlp_model = MLP(
    input_dim=784, output_dim=10, hidden_dims=[30],
    activation_functions=[sigmoid],
    init_parameters_sd=1
)
print(mlp_model)
print()


limit = 3
i = 1
for x, y in zip(x_train, y_train):
    y_hat = mlp_model(x)
    print(f'y_real:\n{y}')
    print(f'\ny_hat:\n{y_hat}')
    i += 1
    if i > limit:
        break
