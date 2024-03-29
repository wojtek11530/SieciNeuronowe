from dataset.and_dataset import get_dataset
from functions.activation_functions import bipolar_activation, unipolar_activation
from models.simple_models.evaluate_model import evaluate_model
from models.simple_models.perceptron import Perceptron
from models.simple_models.train_model import train_model


def run_training(weight_limit: float, unipolar: bool = True):
    learning_rate = 0.01
    dataset = get_dataset(noise_data_number=10, unipolar=unipolar)
    if unipolar:
        print('UNIPOLAR training')
    else:
        print('BIPOLAR training')
    if unipolar:
        perceptron = Perceptron(2, weight_limit=weight_limit, activation_fn=unipolar_activation)
    else:
        perceptron = Perceptron(2, weight_limit=weight_limit, activation_fn=bipolar_activation)
    train_model(perceptron, dataset, learning_rate)
    evaluate_model(perceptron, get_dataset(noise_data_number=2, unipolar=unipolar), unipolar=unipolar)


if __name__ == '__main__':
    weight_lim = 1.0
    run_training(weight_lim, unipolar=True)
    run_training(weight_lim, unipolar=False)
