import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import string
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer, Flatten
import tensorflow_hub as hub


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


def get_data(target_size, color_mode):
    train_dir = "../../data/Train"
    test_dir = "../../data/Test"

    _bs = 128

    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

    _train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=_bs,
        class_mode="categorical",
        color_mode=color_mode,
        subset="training"
    )

    _validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=_bs,
        class_mode="categorical",
        color_mode=color_mode,
        subset="validation"
    )

    _test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=_bs,
        class_mode="categorical",
        color_mode=color_mode
    )

    _classes = [char for char in string.ascii_uppercase if char not in ("J", "Z")]

    return _classes, _bs, _train_generator, _validation_generator, _test_generator


def hub_architecture(input_shape, n_clases, url_model):
    print("HUB")
    model_hub = Sequential(InputLayer(input_shape=input_shape))
    # Una secuencia nueva
    model_hub.add(hub.KerasLayer(url_model, trainable=False))

    model_hub.add(Flatten())
    model_hub.add(Dense(128, activation="relu"))
    model_hub.add(Dropout(0.2))
    model_hub.add(Dense(n_clases, activation="sotfmax"))
    print("BUILD")
    # Una secuencia nueva por usar HUB de TensorFlow
    model_hub.build((None, ) + input_shape)

    print(model_hub.summary())
    return model_hub


def fit_model(classes_, batch_size_, train_generator_, validation_generator_, test_generator_):

    url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5"

    h_model = hub_architecture(input_shape=(150, 150, 3), n_clases=len(classes_), url_model=url)

    h_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    history = h_model.fit(train_generator_, epochs=5, validation_data=validation_generator_, batch_size=batch_size_)

    plot_results(history, "accuracy", "hub results.png")

    h_model.evaluate(test_generator_)


if __name__ == '__main__':

    classes, batch_size, train_generator, validation_generator, test_generator = get_data(target_size=(150, 150),
                                                                                          color_mode="rgb")

    fit_model(classes, batch_size, train_generator, validation_generator, test_generator)

