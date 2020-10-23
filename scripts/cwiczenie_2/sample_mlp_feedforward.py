import numpy as np

from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid, softmax
from models.mlp import MLP

training_data, validation_data, test_data = load_data_wrapper()

mlp_model = MLP(
    input_dim=784, output_dim=10, hidden_dims=[30],
    activation_functions=[sigmoid],
    init_parameters_sd=1
)
print(mlp_model)
print()

x, y = list(zip(*training_data))
x_data = np.array(x)
y_data = np.array(y)

limit = 3
i = 1
for x, y in zip(x_data, y_data):
    y_hat, _ = mlp_model(x)
    print(f'y_real:\n{y}')
    print(f'\ny_hat:\n{y_hat}')
    i += 1
    if i > limit:
        break
