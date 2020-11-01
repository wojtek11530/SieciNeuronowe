from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid
from models.neural_network_models.mlp import MLP
from models.neural_network_models.train_model import train_model, plot_losses_during_training, plot_accuracies


def run_training():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    mlp_model = MLP(
        input_dim=784, output_dim=10, hidden_dims=[30],
        activation_functions=[sigmoid],
        init_parameters_sd=1
    )
    print(mlp_model)
    learning_rate = 1e-1
    batch_size = 50
    max_epochs = 15

    train_model(mlp_model, x_train, y_train, lr=learning_rate, batch_size=batch_size, max_epochs=max_epochs,
                x_val=x_val, y_val=y_val, plot=True, early_stop=True, patience=2)


if __name__ == '__main__':
    run_training()
