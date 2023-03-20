import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import string


def plot_images(images_arr, title):
    fig, axes = plt.subplots(1, 5, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img[:, :, 0], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{title}.png")


if __name__ == '__main__':
    train_dir = "../../data/Train"
    test_dir = "../../data/Test"

    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale",
        subset="training"
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation"
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale"
    )

    classes = [char for char in string.ascii_uppercase if char not in ("J", "Z")]
    print("Classes:", classes)

    sample_training_images, _ = next(train_generator)
    plot_images(sample_training_images[:5], "Train Example")
