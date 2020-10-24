from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid
from models.neural_network_models.mlp import MLP
from models.neural_network_models.train_model import train_model

x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

mlp_model = MLP(
    input_dim=784, output_dim=10, hidden_dims=[30],
    activation_functions=[sigmoid, sigmoid],
    init_parameters_sd=1
)

learning_rate = 1e-2
print(mlp_model)
print()
batch_size = 50
max_epochs = 4

train_model(mlp_model, x_train, y_train, lr=learning_rate, batch_size=batch_size, max_epochs=max_epochs,
            x_val=x_val, y_val=y_val)
