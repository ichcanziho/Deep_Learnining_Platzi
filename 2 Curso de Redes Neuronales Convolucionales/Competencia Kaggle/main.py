import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt


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
    ax1.plot(epoch, loss_values, '-r', label='training')
    ax1.plot(epoch, val_loss_values, '-', label='validation')
    ax2.plot(epoch, metric_values, '-r', label='training')
    ax2.plot(epoch, val_metric_values, '-', label='validation')
    ax1.legend()
    ax2.legend()
    plt.savefig(f"imgs/{fname}")
    plt.close()


def architecture(n_filters, input_shape):
    model = Sequential()

    model.add(Conv2D(filters=n_filters, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=2*n_filters, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=4 * n_filters, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=4 * n_filters, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()

    return model


if __name__ == '__main__':

    train_dategen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    bs = 32

    train_generator = train_dategen.flow_from_directory(directory="data/train", target_size=(150, 150), batch_size=bs,
                                                        class_mode="binary")

    validation_generator = test_datagen.flow_from_directory(directory="data/validation", target_size=(150, 150),
                                                            batch_size=bs, class_mode="binary")

    test_generator = test_datagen.flow_from_directory(directory="data/test", target_size=(150, 150),
                                                      batch_size=bs, class_mode="binary")

    checkpoint_cb = ModelCheckpoint("models/cats_vs_dogs.h5", verbose=1, save_best_only=True, monitor="val_acc")

    md = architecture(n_filters=32, input_shape=(150, 150, 3))
    md.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["acc"])

    history = md.fit(train_generator, steps_per_epoch=2000//bs, epochs=100, validation_data=validation_generator,
                     validation_steps=1000//bs, callbacks=[checkpoint_cb], batch_size=64)

    plot_results(history, "acc", "primeros_resultados.png")

    best_model = load_model("models/cats_vs_dogs.h5")

    acc = best_model.evaluate(test_generator)
    print(acc)

