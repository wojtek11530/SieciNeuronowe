from dataset.and_dataset import get_dataset
from functions.evaluate_model import evaluate_model
from functions.tran_model import train_model
from models.perceptron import Perceptron


def run():
    learning_rate = 0.01
    perceptron = Perceptron(2, weight_limit=1)
    dataset = get_dataset(case_gen_data_number=5)
    train_model(perceptron, dataset, learning_rate)
    evaluate_model(perceptron, get_dataset(case_gen_data_number=2))


if __name__ == '__main__':
    run()
