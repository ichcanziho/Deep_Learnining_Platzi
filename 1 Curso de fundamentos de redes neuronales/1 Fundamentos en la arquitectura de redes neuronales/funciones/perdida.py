import numpy as np


def mse(y: np.array, y_hat: np.array, derivative: bool = False):
    if derivative:
        return y_hat - y
    else:
        return np.mean((y_hat - y)**2)


if __name__ == '__main__':
    real = np.array([0, 0, 1, 1])
    prediction = np.array([0.9, 0.5, 0.2, 0])
    print(mse(real, prediction))
