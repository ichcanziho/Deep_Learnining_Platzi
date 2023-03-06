import numpy as np
import matplotlib.pyplot as plt


def plot_function(x, ys, name):
    plt.plot(x, ys)
    plt.title(name)
    plt.savefig(f"imgs/{name}.png")
    plt.close()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step(x):
    return np.piecewise(x, [x < 0.0, x > 0.0], [0, 1])


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


if __name__ == '__main__':
    x = np.linspace(10, -10, 100)
    plot_function(x, sigmoid(x), "Sigmoid")
    plot_function(x, step(x), "Step")
    plot_function(x, relu(x), "ReLu")
    plot_function(x, tanh(x), "Tanh")
