import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
# Nuevas bibliotecas
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from os.path import exists


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
    model.add(BatchNormalization())
    # Conv 2
    model.add(Conv2D(filters=base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Conv 3
    model.add(Conv2D(filters=2 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # Conv 4
    model.add(Conv2D(filters=2 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Conv 5
    model.add(Conv2D(filters=4 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    # Conv 6
    model.add(Conv2D(filters=4 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
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
    # Limpieza de datos

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

    # Normalizado de datos
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_valid = x_valid.astype('float32')

    mean_train = np.mean(x_train)
    std_train = np.std(x_train)

    x_train = (x_train - mean_train) / (std_train + 1e-7)
    x_test = (x_test - mean_train) / (std_train + 1e-7)
    x_valid = (x_valid - mean_train) / (std_train + 1e-7)

    # Data Augmentation
    datagen = ImageDataGenerator(rotation_range=15,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=True)

    # Checkpoint Callback
    checkpoint_cb = ModelCheckpoint("models/best_model.h5", verbose=1, save_best_only=True, monitor="val_acc")

    if not exists("models/best_model.h5"):
        print("Entrenando por primera vez")
        print("*" * 64)
        print("Creando arquitectura")
        md = architecture(base_filtros=32, w_regularized=1e-4, shape=x_train[0].shape, num_classes=num_clases)
        print("*" * 64)
        print("Empezando a entrenar")
        md.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["acc"])
        history = md.fit(datagen.flow(x_train, y_train, batch_size=128),
                         callbacks=[checkpoint_cb],
                         steps_per_epoch=x_train.shape[0] // 128,
                         epochs=50,
                         verbose=2,
                         validation_data=(x_valid, y_valid))
        plot_results(history, "acc", "resultado_tuneado.png")

    else:
        print("Abriendo modelo y continuando entrenamiento")
        md = load_model("models/best_model.h5")
        history = md.fit(datagen.flow(x_train, y_train, batch_size=128),
                         callbacks=[checkpoint_cb],
                         steps_per_epoch=x_train.shape[0] // 128,
                         epochs=20,
                         verbose=2,
                         validation_data=(x_valid, y_valid))

        plot_results(history, "acc", "resultado_tuneado_2.png")

    acc = md.evaluate(x_test, y_test)
    print("normal acc: ", acc)
    best_model = load_model("models/best_model.h5")
    acc = best_model.evaluate(x_test, y_test)
    print("loaded model acc:", acc)
