from core.utils import get_datasets, plot_results
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import Model
from keras.layers import Dense, Dropout, Input, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model


def load_pretrained(shape, premodel):
    pretrained = premodel(include_top=False, input_tensor=Input(shape=shape))
    for layer in pretrained.layers:
        layer.trainable = False
    last_layers = pretrained.get_layer("mixed7")

    return pretrained, last_layers.output


def functional_architecture(first_model, input_shape, n_clases):
    pretrained, last_output = load_pretrained(shape=input_shape, premodel=first_model)
    new_architecture = Flatten()(last_output)
    new_architecture = Dense(128, activation="relu")(new_architecture)
    new_architecture = Dropout(0.2)(new_architecture)
    new_architecture = Dense(n_clases, activation="softmax")(new_architecture)
    model = Model(pretrained.input, new_architecture)
    return model


def fit_model(classes_, batch_size_, train_generator_, validation_generator_, test_generator_):

    functional_model = functional_architecture(first_model=InceptionV3, input_shape=(150, 150, 3), n_clases=len(classes_))

    functional_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath="models/best_model.h5", frecuency="epoch", save_weights_only=False,
                                 monitor="val_accuracy", save_best_only=True, verbose=1)

    stopping = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1, mode="auto")

    history = functional_model.fit(train_generator_, epochs=50, validation_data=validation_generator_,
                                   batch_size=batch_size_, callbacks=[checkpoint, stopping])

    plot_results(history, "accuracy", "pretrained results.png")

    functional_model.evaluate(test_generator_)

    loaded = load_model("models/best_model.h5")
    loaded.evaluate(test_generator_)


if __name__ == '__main__':
    classes, batch_size, train_generator, validation_generator, test_generator = get_datasets(target_size=(150, 150),
                                                                                              color_mode="rgb")

    fit_model(classes, batch_size, train_generator, validation_generator, test_generator)
