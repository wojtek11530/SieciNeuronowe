def unipolar_loss_function(y_real: float, y_pred: float) -> float:
    return y_real - y_pred


def bipolar_loss_function(y_real: float, y_pred: float) -> float:
    if y_real == 0:
        y_real = -1
    return y_real - y_pred
