import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import reuters
from keras import layers, models
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import regularizers

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def one_hot_encoding(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1
    return results


x_train = one_hot_encoding(train_data)
x_test = one_hot_encoding(test_data)

# Aquí hay un cambio interesante respecto al ejemplo anterior:
# Observemos como luce una etiqueta de y_train

print(train_labels[0], train_labels[0].shape)

# Debemos transformar esta salida en una salida de clasificación multiple, esto es lo mismo
# que hicimos en el problema de clasificación de números escritos a mano del dataset MNIST
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print(y_train[0], y_train[0].shape)


def architecture(model: models.Sequential, input_shape: tuple, n_classes: int) -> models.Sequential:
    model.add(layers.Dense(128, activation="relu", input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    # IMPORTANTE: Ahora que nuestro problema es de clasificación MULTIPLE nuestra activación de la capa de predicción
    # Es diferente, en este caso usamos softmax, porque nos interesa tener la probabilidad de cada clase a la salida.
    model.add(layers.Dense(n_classes, activation="softmax"))
    return model


model_norm = models.Sequential()
model_norm = architecture(model=model_norm, input_shape=(10000, ), n_classes=46)
# Dado que nuestro problema tiene varias clases, entonces usaremos "categorical_crossentropy"
# en lugar de "binary_crossentropy"
model_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

history_norm = model_norm.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.3)

print()
results = model_norm.evaluate(x_test, y_test)
print(results)
history_dict = history_norm.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict["acc"]
val_acc_values = history_dict["val_acc"]
epoch = range(1, len(loss_values) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
fig.suptitle("Neural Network's Result")
ax1.set_title("Loss function over epoch")
ax2.set_title("Acc over epoch")
ax1.set(ylabel="loss", xlabel="epochs")
ax2.set(ylabel="acc", xlabel="epochs")
ax1.plot(epoch, loss_values, 'o-r', label='training')
ax1.plot(epoch, val_loss_values, '--', label='validation')
ax2.plot(epoch, acc_values, 'o-r', label='training')
ax2.plot(epoch, val_acc_values, '--', label='validation')
ax1.legend()
ax2.legend()
plt.savefig("imgs/results.png")
plt.close()
