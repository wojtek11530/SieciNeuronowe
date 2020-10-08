def unipolar_activation(z: float) -> int:
    return 1 if z > 0 else 0


def bipolar_activation(z: float) -> int:
    return 1 if z > 0 else -1
