from datetime import datetime

from dataset.mnist_dataset import load_data_wrapper
from functions.activation_functions import sigmoid
from models.neural_network_models.evaluate_model import evaluate_model
from models.neural_network_models.mlp import MLP
from models.neural_network_models.train_model import train_model
from optimizers.sgd import SGD


def run_training_and_evaluation():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    hidden_dims = [100]
    activation_functions = [sigmoid, sigmoid]
    init_parameters_sd = 1
    learning_rate = 2e-1
    batch_size = 50
    max_epochs = 20

    mlp_model = MLP(
        input_dim=784, output_dim=10, hidden_dims=hidden_dims,
        activation_functions=activation_functions,
        init_parameters_sd=init_parameters_sd,
        optimizer=SGD(learning_rate=learning_rate)
    )
    print(mlp_model)

    train_model(mlp_model, x_train, y_train, batch_size=batch_size, max_epochs=max_epochs,
                x_val=x_val, y_val=y_val, plot=True, early_stop=True, patience=2)

    file_name = f'mlp_model_{hidden_dims}_sd={init_parameters_sd}' + \
                f'_lr={learning_rate}_b={batch_size}_{datetime.now().strftime("%m-%d-%Y_%H.%M")}.pkl'
    mlp_model.save_model(file_name)
    evaluate_model(mlp_model, x_test, y_test)


def evaluate_from_file():
    file_name = 'mlp_model_[100]_sd=1_lr=0.2_b=50_11-01-2020_12.50.pkl'
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_wrapper()

    hidden_dims = [100]
    activation_functions = [sigmoid]
    init_parameters_sd = 1
    mlp_model = MLP(
        input_dim=784, output_dim=10, hidden_dims=hidden_dims,
        activation_functions=activation_functions,
        init_parameters_sd=init_parameters_sd
    )
    print(mlp_model)
    mlp_model.load_model(file_name)
    evaluate_model(mlp_model, x_test, y_test)


if __name__ == '__main__':
    run_training_and_evaluation()
    # evaluate_from_file()
