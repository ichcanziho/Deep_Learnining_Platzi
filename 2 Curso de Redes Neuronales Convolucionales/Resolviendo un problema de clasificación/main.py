import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


def plot_results(history_, metric, fname):
    history_dict = history_.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    metric_values = history_dict[metric]
    val_metric_values = history_dict[f"val_{metric}"]
    epoch = range(1, len(loss_values) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
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
    plt.savefig(f"imgs/{fname}")
    plt.close()


def architecture(base_filtros: int, w_regularized: float, shape: tuple, num_classes: int):
    """
    Definiendo la arquitectura de nuestra CNN
    :param base_filtros: Número de filtros que tomara como base la CNN (capas posteriores usaran multiplos de este número)
    :param w_regularized: Peso para utilizar por el regularizador L2
    :param shape: forma del tensor de entrada (dimensiones de las imágenes de entrenamiento)
    :param num_classes: número de clases a clasificar por la CNN
    :return:
    """
    model = Sequential()

    # Conv 1
    model.add(Conv2D(filters=base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized), input_shape=shape))
    model.add(Activation("relu"))
    # Conv 2
    model.add(Conv2D(filters=base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Conv 3
    model.add(Conv2D(filters=2*base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    # Conv 4
    model.add(Conv2D(filters=2 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Conv 5
    model.add(Conv2D(filters=4 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    # Conv 6
    model.add(Conv2D(filters=4 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    # Flatten
    model.add(Flatten())
    # Capa de clasificación
    model.add(Dense(units=num_classes, activation="softmax"))
    print(model.summary())
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Análisis exploratorio
    print("x_train shape:", x_train.shape)
    plt.imshow(x_train[0])
    plt.savefig("imgs/train0.jpg")
    plt.close()
    # Limpieza de datos
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    num_clases = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_clases)
    y_test = to_categorical(y_test, num_clases)

    # Creando nuevas particiones de los datos
    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]
    print('x_train shape', x_train.shape)
    print('x_train shape [0]', x_train[0].shape)

    print('train:', x_train.shape[0])
    print('val:', x_valid.shape[0])
    print('test:', x_test.shape[0])
    print("*"*64)
    print("Creando arquitectura")
    md = architecture(base_filtros=32, w_regularized=1e-4, shape=x_train[0].shape, num_classes=num_clases)
    print("*"*64)
    print("Empezando a entrenar")
    md.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
    history = md.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_valid, y_valid), shuffle=True,)

    plot_results(history, "acc", "primer_resultado.png")

    md.evaluate(x_test, y_test)
