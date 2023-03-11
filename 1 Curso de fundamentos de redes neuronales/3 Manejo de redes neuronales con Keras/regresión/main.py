import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.datasets import boston_housing
from keras import models, layers
# Biblioteca para implementar K-Fold CrossValidation
from sklearn.model_selection import KFold
# Biblioteca para la correcta normalización de datos numéricos
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def plot_results(out, metric, offset):
    print(out)
    df = {}
    for key in out[0].keys():
        row = []
        for fold in out:
            row.append(fold[key])
        row = np.array(row).mean(axis=0)
        df[key] = row
    frame = pd.DataFrame(df)
    frame = frame[offset:]
    print(frame)
    loss_values = frame['loss']
    val_loss_values = frame['val_loss']
    metric_values = frame[metric]
    val_metric_values = frame[f"val_{metric}"]
    epoch = range(1, len(loss_values) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle("Neural Network's Result")
    ax1.set_title("Loss function over epoch")
    ax2.set_title(f"{metric} over epoch")
    ax1.set(ylabel="loss", xlabel="epochs")
    ax2.set(ylabel=metric, xlabel="epochs")
    ax1.plot(epoch, loss_values, 'o-r', label='training')
    ax1.plot(epoch, val_loss_values, '--', label='validation')
    ax2.plot(epoch, metric_values, 'o-r', label='training')
    ax2.plot(epoch, val_metric_values, '--', label='validation')
    ax1.legend()
    ax2.legend()
    plt.savefig("imgs/results.png")
    plt.close()


def train_test_split_kf(xs: np.array, ys: np.array, train_size: np.array, test_size: np.array) -> np.array:
    x_train_ = xs[train_size]
    x_test_ = xs[test_size]
    y_train_ = ys[train_size]
    y_test_ = ys[test_size]
    return x_train_, x_test_, y_train_, y_test_


def build_model_regression(dim):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=dim))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    # Como la última capa es una predicción de regresión, NO necesita una capa de activación
    model.add(layers.Dense(1))
    # El error sí será el mean squared error, pero la métrica debe ser diferente, en este caso max absolute error
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


if __name__ == '__main__':
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    print(train_data[0])
    print(train_targets[0])

    # Implementando cross validation:
    n_epochs = 40
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    standard_scaler = StandardScaler()
    all_history = []  # Aquí guardaremos los resultados de cada fold
    for n_fold, (train, test) in enumerate(kf.split(train_data)):
        print(f"\t-I'm running fold {n_fold + 1}")
        x_train, x_test, y_train, y_test = train_test_split_kf(xs=train_data, ys=train_targets,
                                                               train_size=train, test_size=test)
        standard_scaler.fit(x_train)
        x_train_s = standard_scaler.transform(x_train)
        # Es MUY importante tener en cuenta que los datos de prueba NO LOS CONOZCO entonces NO tiene sentido obtener
        # el promedio y desviación standard de una muestra que NO conozco por eso estoy normalizando con el promedio
        # y std de la muestra a la que mi modelo sí tenía acceso mientras fue entrenada.
        x_test_s = standard_scaler.transform(x_test)

        model = build_model_regression(dim=13)
        history = model.fit(x_train_s, y_train, epochs=n_epochs, batch_size=16,
                            validation_data=(x_test_s, y_test), verbose=0)
        all_history.append(history.history)
        print("*"*64)

    print("End CrossValidation process.")

    plot_results(all_history, "mae", offset=5)
    results = model.evaluate(test_data, test_targets)
    print(results)
