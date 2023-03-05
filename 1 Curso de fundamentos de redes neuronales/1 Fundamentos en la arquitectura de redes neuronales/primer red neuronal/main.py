# Estas librerias solo son necesario importarlas porque estoy corriendo de forma local el código
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Empiezan las librerias que vamos a utilizar para crear nuestro modelo de DEEP LEARNING
from keras import layers, models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

"""
En esta clase nosotros aprenderemos a:
    1: Utilizar datasets pre-cargados de Keras
    2: Familiarizarnos con los shapes de los datos de entrenamiento y validación de un dataset de DL
    3: Crear un simple Modelo Secuencial de 1 capa y Multiples Salidas
    4: Compilar el modelo con ciertos parámetros
    5: Ver el resumen de la configuración de nuestro Modelo
    6: Modificar los datos de entrenamiento y validación para hacerlos más manejables por el modelo de DL
    7: Entrenar a nuestra red neuronal
    8: Evaluar el rendimiento de nuestra red neuronal con datos de validación
    9: Guardar el modelo para usarlo después
"""
# ----------------------------------------------------------------------------------------------------------------------
#        1: Utilizar datasets pre-cargados de Keras
# ----------------------------------------------------------------------------------------------------------------------
print("*"*64, 1, "*"*64)
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# ----------------------------------------------------------------------------------------------------------------------
#        2: Familiarizarnos con los shapes de los datos de entrenamiento y validación de un dataset de DL
# ----------------------------------------------------------------------------------------------------------------------
print("*"*64, 2, "*"*64)

print("Train data shape:", train_data.shape)
print("Train data example shape:", train_data[0].shape)
print("Train label shape:", train_labels.shape)
print("Train label example shape:", train_labels[0].shape)
print("Test data shape:", test_data.shape)
print("Test data example shape:", test_data[0].shape)
print("Test label shape:", test_labels.shape)
print("Test label example shape:", test_labels[0].shape)

plt.imshow(train_data[0])
plt.savefig("outputs/numero.png")
print(train_labels[0])

# ----------------------------------------------------------------------------------------------------------------------
#        3: Crear un simple Modelo Secuencial de 1 capa y Multiples Salidas
# ----------------------------------------------------------------------------------------------------------------------
print("*"*64, 3, "*"*64)

model = models.Sequential()
model.add(layers.Dense(512, activation="relu", input_shape=(28*28, )))
model.add(layers.Dense(10, activation="softmax"))

# ----------------------------------------------------------------------------------------------------------------------
#        4: Compilar el modelo con ciertos parámetros
# ----------------------------------------------------------------------------------------------------------------------
print("*"*64, 4, "*"*64)

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics="accuracy")

# ----------------------------------------------------------------------------------------------------------------------
#        5: Ver el resumen de la configuración de nuestro Modelo
# ----------------------------------------------------------------------------------------------------------------------
print("*"*64, 5, "*"*64)

print(model.summary())

# ----------------------------------------------------------------------------------------------------------------------
#        6: Modificar los datos de entrenamiento y validación para hacerlos más manejables por el modelo de DL
# ----------------------------------------------------------------------------------------------------------------------
print("*"*64, 6, "*"*64)

x_train = train_data.reshape((60000, 28*28))
x_train = x_train.astype("float32")/255

x_test = test_data.reshape((10000, 28*28))
x_test = x_test.astype("float32")/255

print(x_train[0].shape)
print(x_train[0])

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print(y_train[0])

# ----------------------------------------------------------------------------------------------------------------------
#        7: Entrenar a nuestra red neuronal
# ----------------------------------------------------------------------------------------------------------------------
print("*"*64, 7, "*"*64)

model.fit(x_train, y_train, epochs=5, batch_size=128)

# ----------------------------------------------------------------------------------------------------------------------
#        8: Evaluar el rendimiento de nuestra red neuronal con datos de validación
# ----------------------------------------------------------------------------------------------------------------------
print("*"*64, 8, "*"*64)

model.evaluate(x_test, y_test)

# ----------------------------------------------------------------------------------------------------------------------
#        9: Guardar el modelo para usarlo después
# ----------------------------------------------------------------------------------------------------------------------
print("*"*64, 9, "*"*64)

model.save("Model/numeros.h5")

