import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import string
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.regularizers import l2
import json
from keras import Model


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


def conv_architecture(input_shape, n_clases):
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu", kernel_regularizer=l2(1e-5)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(1e-5)))
    model.add(Dropout(0.2))
    model.add(Dense(n_clases, activation="softmax"))
    print(model.summary())
    return model


def generate_weights(classes, batch_size, train_generator, validation_generator, test_generator):

    conv_model = conv_architecture(input_shape=(28, 28, 1), n_clases=len(classes))

    config_dict = conv_model.get_config()

    with open('config_model.json', 'w') as outfile:
        json.dump(config_dict, outfile)

    conv_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    #
    conv_model.fit(train_generator, epochs=5, validation_data=validation_generator, batch_size=batch_size)

    conv_model.evaluate(test_generator)

    conv_model.save_weights('only_weights.h5')


def load_model_from_architecture_and_weights(architecture, weights) -> Model:
    json_config = json.load(open(architecture))
    new_model = Sequential.from_config(json_config)
    new_model.load_weights(weights)
    new_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return new_model


if __name__ == '__main__':

    classes, batch_size, train_generator, validation_generator, test_generator = get_data()

    generate_weights(classes, batch_size, train_generator, validation_generator, test_generator)

    loaded_model = load_model_from_architecture_and_weights("config_model.json", "only_weights.h5")
    print("from loaded:")
    loaded_model.evaluate(test_generator)

