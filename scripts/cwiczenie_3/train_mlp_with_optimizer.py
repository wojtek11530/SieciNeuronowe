from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid
from models.neural_network_models.mlp import MLP
from models.neural_network_models.train_model import train_model
from optimizers.momentum import Momentum
from optimizers.nestorov_momentum import NestorovMomentum


def run_training():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    learning_rate = 1e-1
    batch_size = 50
    max_epochs = 8

    mlp_model = MLP(
        input_dim=784, output_dim=10, hidden_dims=[30],
        activation_functions=[sigmoid],
        init_parameters_sd=1,
        optimizer=NestorovMomentum(learning_rate=learning_rate, momentum_rate=0.7)
    )

    print(mlp_model)

    train_model(mlp_model, x_train, y_train, lr=learning_rate, batch_size=batch_size, max_epochs=max_epochs,
                x_val=x_val, y_val=y_val, plot=True)


if __name__ == '__main__':
    run_training()
