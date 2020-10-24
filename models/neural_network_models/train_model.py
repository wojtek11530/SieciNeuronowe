import sys
from typing import Optional, Tuple, List

import numpy as np
from tqdm import tqdm

from models.neural_network_models.neural_network_base import NeuralNetworkBaseModel


def train_model(model: NeuralNetworkBaseModel, x_train: np.ndarray, y_train: np.ndarray,
                lr: float, batch_size: int, max_epochs: int,
                x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) \
        -> Tuple[int, List[float], List[float], List[float]]:
    training_losses = []
    validation_losses = []
    validation_accuracies = []

    overall_epoch_num = max_epochs
    for epoch_num in range(max_epochs):
        batch_losses = []

        randomize = np.arange(len(x_train))
        np.random.shuffle(randomize)
        x_train = x_train[randomize]
        y_train = y_train[randomize]

        with tqdm(range(0, len(x_train), batch_size), desc=f'Epoch {epoch_num}', file=sys.stdout) as t:
            for i in t:
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                _, batch_loss = model.update_weight(x_batch, y_batch, lr=lr)
                t.set_postfix(training_loss=batch_loss)
                batch_losses.append(batch_loss)

            avg_training_loss = float(np.mean(batch_losses))
            training_losses.append(avg_training_loss)

        if x_val is not None and y_val is not None:
            avg_val_loss, accuracy = validate_epoch(model, x_val, y_val, epoch_num)
            validation_losses.append(avg_val_loss)
            validation_accuracies.append(accuracy)

    return overall_epoch_num, training_losses, validation_losses, validation_accuracies


def validate_epoch(model: NeuralNetworkBaseModel, x_val: np.ndarray, y_val: np.ndarray, epoch_num: int) \
        -> Tuple[float, float]:
    y_label = np.argmax(y_val, axis=1).flatten()
    y_pred = np.array([model(x) for x in tqdm(x_val, desc=f'Epoch {epoch_num} validation', file=sys.stdout)])
    y_pred_label = np.argmax(y_pred, axis=1).flatten()
    avg_val_loss = float(np.mean(model.calculate_losses(y_pred, y_val)))
    accuracy = sum(y_label == y_pred_label) / len(y_val)
    print(f'Avg_val_loss={avg_val_loss:.4f}, Val_accuracy={accuracy:.4f}')
    return avg_val_loss, accuracy
