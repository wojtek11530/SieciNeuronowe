import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from models.neural_network_models.neural_network_base import NeuralNetworkBaseModel


def evaluate_model(model: NeuralNetworkBaseModel, x_test: np.ndarray, y_test: np.ndarray):
    print('\nModel evaluation:')
    y_label = np.argmax(y_test, axis=1).flatten()
    y_pred = np.array([model(x) for x in tqdm(x_test, desc='Model evaluation', file=sys.stdout)])
    y_pred_label = np.argmax(y_pred, axis=1).flatten()
    accuracy = sum(y_label == y_pred_label) / len(y_test)
    print(f'Aaccuracy={accuracy:.4f}')

    print(classification_report(y_label, y_pred_label))

    cm = confusion_matrix(y_label, y_pred_label)
    df_cm = pd.DataFrame(cm, index=np.unique(y_label), columns=np.unique(y_label))
    show_confusion_matrix(df_cm)


def show_confusion_matrix(conf_matrix: pd.DataFrame) -> None:
    hmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Predykowana etykieta')
    plt.tight_layout()
    plt.show()
