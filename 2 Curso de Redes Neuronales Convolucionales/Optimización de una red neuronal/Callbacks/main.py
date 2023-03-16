import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from keras.utils import to_categorical
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


def architecture(shape, name):
    model_ = Sequential(name=name)
    model_.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=shape))
    model_.add(MaxPool2D(pool_size=2))
    # model_.add(Dropout(0.3))
    model_.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model_.add(MaxPool2D(pool_size=2))
    # model_.add(Dropout(0.3))
    model_.add(Flatten())
    model_.add(Dense(256, activation="relu"))
    # model_.add(Dropout(0.5))
    model_.add(Dense(10, activation="softmax"))
    print(model_.summary())
    return model_


if __name__ == '__main__':
    # Descargando dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    train_labels_categorical = to_categorical(train_labels, 10)
    test_labels_categorical = to_categorical(test_labels, 10)

    # --------------------------------------------------------
    # --------------- Experimento 1 --------------------------
    # --------------------------------------------------------

    # Callback de Early Stopping
    early_stopping_cb = EarlyStopping(monitor="val_accuracy", patience=1, verbose=1)
    # Creando arquitectura del modelo
    model = architecture(shape=train_images[0].shape, name="Early_Stopping")
    # Compilando el modelo
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.fit(train_images, train_labels_categorical, batch_size=256, epochs=10, validation_split=0.3,
              callbacks=[early_stopping_cb])
    score = model.evaluate(test_images, test_labels_categorical)
    print(score)

    # --------------------------------------------------------
    # --------------- Experimento 2 --------------------------
    # --------------------------------------------------------

    # Callback de Checkpoint
    checkpoint_cb = ModelCheckpoint(filepath="models/best_model.h5", save_weights_only=False, monitor="accuracy",
                                    mode="max",
                                    save_best_only=True, verbose=1)
    # Creando arquitectura del modelo
    model = architecture(shape=train_images[0].shape, name="Checkpoint")
    # Compilando el modelo
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    model.fit(train_images, train_labels_categorical, batch_size=64, epochs=10, validation_split=0.3,
              callbacks=[checkpoint_cb])

    loaded_model = load_model("models/best_model.h5")

    print("model:", model.evaluate(test_images, test_labels_categorical))
    print("loaded model:", loaded_model.evaluate(test_images, test_labels_categorical))

