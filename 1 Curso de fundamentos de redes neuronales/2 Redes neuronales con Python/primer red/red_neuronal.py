import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

"""
En esta clase nosotros aprenderemos a:
    1: Creación de un dataset Artificial
    2: Definimos nuestras funciones de activación
    3: Función de perdida
    4: Función inicializadora de pesos
    5: Forward propagation
    6: Backpropagation
    7: Gradient descent
    8: Train Model Function
    9: Definir arquitectura de la red
    10: Entrenamos el modelo
    11: Probando el modelo sobre datos nuevos
"""


# ----------------------------------------------------------------------------------------------------------------------
#        2: Definimos nuestras funciones de activación
# ----------------------------------------------------------------------------------------------------------------------

def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
    else:
        return 1 / (1 + np.exp(-x))


def relu(x, derivate=False):
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return np.maximum(0, x)


# ----------------------------------------------------------------------------------------------------------------------
#        3: Función de perdida
# ----------------------------------------------------------------------------------------------------------------------
def mse(y, y_hat, derivate=False):
    if derivate:
        return 2*(y_hat - y)
    else:
        return np.mean((y_hat - y) ** 2)


# ----------------------------------------------------------------------------------------------------------------------
#        4: Función Inicializadora de pesos
# ----------------------------------------------------------------------------------------------------------------------

# recibe como parámetro de entrada una lista que contenga la cantidad de neuronas que tendrá cada capa de la red
def initialize_parameters_deep(layer_dims: list) -> dict:
    """
    Genera un diccionario de pesos y sesgos de una red neuronal de acuerdo a su arquitectura de capas
    :param layer_dims: lista que representa la cantidad de neuronas presente en cada capa de la red
    :return: dict: parameters.
    """
    parameters = {}
    L = len(layer_dims)
    for l in range(0, L - 1):
        parameters[f'W{l + 1}'] = (np.random.rand(layer_dims[l],
                                                  layer_dims[l + 1]) * 2) - 1  # Multiplicar por 2 y restar
        # 1 es una forma de normalizar los datos para que vayan de -1 a 1, de esta forma encajan mejor con la
        # distribución de datos de entrada de nuestro problema, pero tampoco es indispensable
        parameters[f'b{l + 1}'] = (np.random.rand(1, layer_dims[l + 1]) * 2) - 1
        print(f"Inicializando PESO W{l + 1} con dimensiones:", parameters[f'W{l + 1}'].shape)
        print(f"Inicializando BIAS b{l + 1} con dimensiones:", parameters[f'b{l + 1}'].shape)

    return parameters


# ----------------------------------------------------------------------------------------------------------------------
#        5: Forward Propagation
# ----------------------------------------------------------------------------------------------------------------------
def linear_forward(A, W, b):
    Z = np.dot(A, W) + b
    return Z


def linear_activation_forward(A_prev, W, b, activation_function):
    Z = linear_forward(A_prev, W, b)
    A = activation_function(Z)
    return A


def forward_step(A0, params, activations_functions, n_layers):
    L = n_layers
    params["A0"] = A0
    for i in range(1, L + 1):
        params[f"A{i}"] = linear_activation_forward(params[f"A{i - 1}"], params[f"W{i}"], params[f"b{i}"],
                                                    activations_functions[i])
    y_hat = params[f"A{L}"]
    return y_hat


# ----------------------------------------------------------------------------------------------------------------------
#        6: Backpropagation
# ----------------------------------------------------------------------------------------------------------------------


def backpropagation(Y, y_hat, params, activations_functions, error_function, n_layers):
    L = n_layers
    params[f'dZ{L}'] = error_function(Y, y_hat, True) * activations_functions[L](params[f'A{L}'], True)
    params[f'dW{L}'] = np.dot(params[f'A{L - 1}'].T, params[f'dZ{L}'])

    for l in reversed(range(2, L + 1)):
        params[f'dZ{l - 1}'] = np.matmul(params[f'dZ{l}'], params[f'W{l}'].T) * activations_functions[l - 1](
            params[f'A{l - 1}'], True)

    for l in reversed(range(1, L)):
        params[f'dW{l}'] = np.matmul(params[f'A{l - 1}'].T, params[f'dZ{l}'])

    return params


# ----------------------------------------------------------------------------------------------------------------------
#        7: Gradient descent
# ----------------------------------------------------------------------------------------------------------------------


def gradient_descent(params, lr, n_layers):
    L = n_layers

    for l in reversed(range(1, L + 1)):
        params[f'W{l}'] = params[f'W{l}'] - params[f'dW{l}'] * lr
        params[f'b{l}'] = params[f'b{l}'] - (np.mean(params[f'dZ{l}'], axis=0, keepdims=True)) * lr

    return params


# ----------------------------------------------------------------------------------------------------------------------
#        8: Train Model Function
# ----------------------------------------------------------------------------------------------------------------------

def train_model(X, Y, layer_dims, params, activations_functions, error_function, lr, epochs):
    errors = []
    n_layers = len(layer_dims) - 1
    j = 1
    for _ in range(epochs):
        y_hat = forward_step(X, params, activations_functions, n_layers)
        params = backpropagation(Y, y_hat, params, activations_functions, error_function, n_layers)
        params = gradient_descent(params, lr, n_layers)

        if _ % 100 == 0:
            e = error_function(Y, y_hat)
            if _ % 1000 == 0:
                print(j, "error:", e)
                j += 1
            errors.append(e)

    return errors, params


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    #        1: Creción de un dataset artificial
    # ------------------------------------------------------------------------------------------------------------------
    N = 1000
    gq = make_gaussian_quantiles(mean=None, cov=0.1, n_samples=N, n_features=2, n_classes=2, shuffle=True,
                                 random_state=21)

    X, Y = gq
    # Esto es necesario para hacer el plot más cómodo
    Y = Y[:, np.newaxis]
    # X son las entradas de mi red, tienen 2 dimensiones, Y son las predicciones que corresponde a 2 clases.
    print(X.shape, Y.shape)
    # Muestro un scatter plot de la distribución de mis datos
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=40)
    plt.title("Problema de clasificación")
    plt.savefig("imgs/clasificacion.png")
    plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    #        9: Definir arquitectura de la red
    # ------------------------------------------------------------------------------------------------------------------

    layer_dims = [2, 4, 8, 1]
    lr = 0.002
    activations_functions = [0, relu, relu, sigmoid]
    params = initialize_parameters_deep(layer_dims)
    epochs = 10000

    # ------------------------------------------------------------------------------------------------------------------
    #        10: Entrenamos el modelo
    # ------------------------------------------------------------------------------------------------------------------

    errors, params = train_model(X, Y, layer_dims, params, activations_functions, mse, lr, epochs)

    plt.plot(errors)
    plt.title("MSE over epochs")
    plt.xlabel("epochs")
    plt.ylabel("MSE")
    plt.savefig("imgs/model.png")
    plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    #        11: Probando el modelo sobre datos nuevos
    # ------------------------------------------------------------------------------------------------------------------

    data_test = (np.random.rand(1000, 2) * 2) - 1
    prediction = forward_step(data_test, params, activations_functions, 3)
    y = np.where(prediction >= 0.5, 1, 0)
    plt.scatter(data_test[:, 0], data_test[:, 1], c=y[:, 0], s=40)
    plt.title("NN prediction")
    plt.savefig("imgs/prediction.png")
    plt.close()
