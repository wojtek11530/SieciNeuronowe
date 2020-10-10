import matplotlib.pyplot as plt

from dataset.and_dataset import get_dataset
from functions.activation_functions import bipolar_activation, unipolar_activation
from functions.tran_model import train_model
from models.perceptron import Perceptron

unipolar_dataset = get_dataset(noise_data_number=5, unipolar=True)
bipolar_dataset = get_dataset(noise_data_number=5, unipolar=False)

simulations_num = 50
lr = 0.01
weight_limit = 0.5

unipolar_epochs_num = []
bipolar_epochs_num = []

for i in range(simulations_num):
    unipolar_perceptron = Perceptron(2, weight_limit=weight_limit, activation_fn=unipolar_activation)
    epoch_num, _ = train_model(unipolar_perceptron, unipolar_dataset, lr, verbose=False)
    unipolar_epochs_num.append(epoch_num)

    unipolar_perceptron = Perceptron(2, weight_limit=weight_limit, activation_fn=bipolar_activation)
    epoch_num, _ = train_model(unipolar_perceptron, bipolar_dataset, lr, verbose=False)
    bipolar_epochs_num.append(epoch_num)

plt.boxplot([unipolar_epochs_num, bipolar_epochs_num], labels=['Aktywacja unipolarna', 'Aktywacja bipolarna'])
plt.ylabel('Liczba epok')
plt.grid(axis='y')
plt.show()
