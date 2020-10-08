from dataset.and_dataset import get_dataset
from functions.activation_functions import bipolar_activation
from functions.evaluate_model import evaluate_model
from functions.loss_functions import bipolar_loss_function
from functions.tran_model import train_model
from models.perceptron import Perceptron


def run():
    learning_rate = 0.1
    perceptron = Perceptron(2, weight_limit=0.1, loss_fn=bipolar_loss_function, activation_fn=bipolar_activation)
    dataset = get_dataset(noise_data_number=0)
    train_model(perceptron, dataset, learning_rate)
    evaluate_model(perceptron, get_dataset(noise_data_number=2))


if __name__ == '__main__':
    run()
