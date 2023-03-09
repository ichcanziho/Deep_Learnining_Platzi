import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import imdb
from keras import models, layers
import numpy as np
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt

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
#        1: Obtención de datos imdb - Keras
# ----------------------------------------------------------------------------------------------------------------------

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Cómo luce uno de los datos de entrenamiento?
print("Train data example")
print(train_data[0])
print("Train label example")
print(train_labels[0])


def convert_number_to_word(example):
    word_index = imdb.get_word_index()
    word_index = dict([(value, key) for (key, value) in word_index.items()])
    print(" ".join([str(word_index.get(_ - 3)) for _ in example]))


# Observemos entonces cómo luce realmente un texto de entrada
convert_number_to_word(train_data[0])


# ----------------------------------------------------------------------------------------------------------------------
#        2: Normalizando datos
# ----------------------------------------------------------------------------------------------------------------------

def one_hot_encoding(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1
    return results


x_train = one_hot_encoding(train_data)
x_test = one_hot_encoding(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(train_data.shape)
print(x_train.shape)
print(x_train[0], x_train[0].shape)


# ----------------------------------------------------------------------------------------------------------------------
#        3: Arquitectura del Modelo
# ----------------------------------------------------------------------------------------------------------------------

def architecture(model: models.Sequential) -> models.Sequential:
    model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


model = models.Sequential()
model = architecture(model)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

# ----------------------------------------------------------------------------------------------------------------------
#        4: Entrenando el Modelo
# ----------------------------------------------------------------------------------------------------------------------

history = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.3)

# ----------------------------------------------------------------------------------------------------------------------
#        5: Analizando resultados
# ----------------------------------------------------------------------------------------------------------------------
print()
results = model.evaluate(x_test, y_test)
print(results)


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

fig = plt.figure(figsize=(10, 10))
epoch = range(1, len(loss_values) + 1)
plt.plot(epoch, loss_values, 'o-r', label='training')
plt.plot(epoch, val_loss_values, '--', label='validation')
plt.title("Error in training and validation datasets")
plt.xlabel("epochs")
plt.ylabel("Binary Cross Entropy")
plt.legend()
plt.savefig("imgs/errores.png")
plt.close()
print()
