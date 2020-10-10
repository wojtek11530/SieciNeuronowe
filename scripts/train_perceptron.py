from dataset.and_dataset import get_dataset
from functions.activation_functions import bipolar_activation, unipolar_activation
from functions.evaluate_model import evaluate_model
from functions.tran_model import train_model
from models.perceptron import Perceptron


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
    weight_lim = 0.1
    run_training(weight_lim, unipolar=True)
    run_training(weight_lim, unipolar=False)
