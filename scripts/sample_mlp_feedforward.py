from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid, softmax
from models.mlp import MLP

training_data, validation_data, test_data = load_data_wrapper()

mlp_model = MLP(
    input_dim=784, output_dim=10, hidden_dims=[30],
    activation_functions=[sigmoid, softmax],
    init_parameters_sd=1
)
print(mlp_model)

for x, y in training_data:
    y_hat = mlp_model(x)
    print(f'y_real:\n{y}')
    print(f'y_hat:\n{y_hat}')
    break
