import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from keras.utils import to_categorical
from keras import Sequential
from keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt


def architecture(model_: Sequential):
    model_.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1)))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    model_.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    # Esta capa sirve para aplanar y pasar de redes convolucionales a normales
    model_.add(Flatten())
    model_.add(Dense(256, activation="relu"))
    model_.add(Dropout(0.5))
    # Como es un problema de clasificación multilabel usamos softmax como activación de la última capa
    model_.add(Dense(10, activation="softmax"))
    print(model_.summary())
    # Compilamos el modelo con la información que YA conocemos (la última capa de la red CNN es igual a las que ya hemos
    # trabajo anteriormente)
    model_.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return model_


def architecture_sparse(model_: Sequential):
    model_.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1)))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    model_.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    model_.add(Flatten())
    model_.add(Dense(256, activation="relu"))
    model_.add(Dropout(0.5))
    # Utilizando como perdida la SparceCategoricalCrossentropy NO es necesario usar "Softmax" como activación
    model_.add(Dense(10))
    print(model_.summary())
    model_.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer="rmsprop", metrics=["accuracy"])
    return model_


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


if __name__ == '__main__':
    # Descargando dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # Análisis exploratorio
    print(train_images.shape)
    plt.imshow(train_images[0])
    plt.savefig("imgs/train0.jpg")
    plt.close()
    # Normalizado de imágenes
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    # a diferencia de las redes neuronales normales, dónde la entrada debía ser un vector de 1 dim
    # en las CNN la entrada es una matriz, es por eso que en el reshape debemos tomar en cuenta
    # [[]].reshape(n, x, y, c)
    # n, x, y, c -> n = número de imágenes, x = ancho de la imagen, y = largo de la imagen, c = número de canales
    # Dado que nuestras imágenes están en escala de grises, entonces el número de canales que maneja es 1.
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    # CONOCIMIENTO del curso anterior -> Transformando números del 0 al 9 (10 clases) en su One Hot Encoding
    train_labels_categorical = to_categorical(train_labels, 10)
    test_labels_categorical = to_categorical(test_labels, 10)

    model = Sequential()
    model = architecture(model)
    history = model.fit(train_images, train_labels_categorical, batch_size=64, epochs=10, validation_split=0.3)
    score = model.evaluate(test_images, test_labels_categorical)
    print(score)
    plot_results(history, "accuracy", "results_base.png")

    print("="*64)
    # BONUS -> Creando una arquitectura que NO necesite usar softmax para clasificar entre las 10 clases

    model = Sequential()
    model = architecture_sparse(model)
    # Este tipo de modelo NO me exige usar las etiquetas como categóricas, por eso puedo usar train_labels normal.
    history = model.fit(train_images, train_labels, batch_size=64, epochs=10, validation_split=0.3)
    score = model.evaluate(test_images, test_labels)
    print(score)
    plot_results(history, "accuracy", "results_sparse.png")
