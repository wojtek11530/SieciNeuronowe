from dataset.and_dataset import get_dataset
from functions.evaluate_model import evaluate_model
from functions.tran_model import train_model
from models.adaline import Adaline


def run_training(weight_limit: float):
    learning_rate = 1e-2
    error_margin = 1e-5
    dataset = get_dataset(noise_data_number=0, unipolar=False)

    adaline = Adaline(2, weight_limit=weight_limit)
    train_model(adaline, dataset, learning_rate, error_margin=error_margin)
    evaluate_model(adaline, get_dataset(noise_data_number=2, unipolar=False), unipolar=False)


if __name__ == '__main__':
    run_training(weight_limit=0.1)
