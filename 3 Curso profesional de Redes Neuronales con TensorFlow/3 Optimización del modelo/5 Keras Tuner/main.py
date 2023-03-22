import os
from abc import ABC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import string
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras_tuner import HyperModel, Hyperband
import json


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
    ax1.plot(epoch, loss_values, 'go-', label='training')
    ax1.plot(epoch, val_loss_values, 'ro-', label='validation')
    ax2.plot(epoch, metric_values, 'go-', label='training')
    ax2.plot(epoch, val_metric_values, 'ro-', label='validation')
    ax1.legend()
    ax2.legend()
    plt.savefig(f"{fname}")
    plt.close()


def get_data():
    train_dir = "../../data/Train"
    test_dir = "../../data/Test"

    _bs = 128

    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

    _train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        batch_size=_bs,
        class_mode="categorical",
        color_mode="grayscale",
        subset="training"
    )

    _validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=_bs,
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation"
    )

    _test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=_bs,
        class_mode="categorical",
        color_mode="grayscale"
    )

    _classes = [char for char in string.ascii_uppercase if char not in ("J", "Z")]

    return _classes, _bs, _train_generator, _validation_generator, _test_generator


class CNNArchitecture(HyperModel, ABC):

    def __init__(self, input_shape, n_classes):
        super().__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes

    def build(self, hp):
        model = Sequential()

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

        model.add(Dense(units=hp.Int("units_1", min_value=64, max_value=512, step=64, default=128),
                        activation="relu", kernel_regularizer=l2(1e-5)))
        model.add(Dropout(rate=hp.Float("dropout_1", min_value=0.2, max_value=0.6, default=0.5, step=0.10)))
        model.add(BatchNormalization())

        model.add(Dense(128, activation="relu", kernel_regularizer=l2(1e-5)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(self.n_classes, activation="softmax"))

        model.compile(Adam(hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3,)),
                      loss="categorical_crossentropy", metrics=['accuracy'])
        # Alternativa a hp.Float -> hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
        return model


if __name__ == '__main__':

    print("Creando particiones de datasets")
    print("="*64)
    classes, batch_size, train_generator, validation_generator, test_generator = get_data()

    print("Creando Arquitectura del hyper-modelo")
    print("=" * 64)
    cnn_model = CNNArchitecture(input_shape=(28, 28, 1), n_classes=len(classes))

    print("Buscando mejor configuración")
    print("=" * 64)
    tuner = Hyperband(hypermodel=cnn_model, objective="val_accuracy", max_epochs=20, factor=3, directory="models/",
                      project_name="test")
    tuner.search(train_generator, epochs=20, validation_data=validation_generator)

    print("Resultados obtenidos")
    print("=" * 64)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps)

    conv_model = tuner.hypermodel.build(best_hps)

    print("Guardando configuración")
    print("=" * 64)
    config_dict = conv_model.get_config()

    print(config_dict)

    with open('config_model.json', 'w') as outfile:
        json.dump(config_dict, outfile)

    callback = EarlyStopping(monitor="val_accuracy", patience=3, mode="auto")

    checkpoint = ModelCheckpoint(filepath="models/best_model.h5", save_best_only=True, save_weights_only=False,
                                 mode="auto", verbose=1, monitor="val_accuracy")

    print("Entrenando al mejor modelo")
    print("=" * 64)
    history = conv_model.fit(train_generator, epochs=20, validation_data=validation_generator, batch_size=128,
                             callbacks=[callback, checkpoint])

    plot_results(history, "accuracy", "hypermodel_results.png")

    results = conv_model.evaluate(test_generator)
