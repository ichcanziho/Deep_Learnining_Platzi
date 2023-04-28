import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from architecture import create_unet
import tensorflow as tf
from datetime import date
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from utils import read_dataset, get_partitions, load_data, plot_results, predict_test_samples, plot_images


def create_and_compile_model(train_samples, test_samples):
    inputs = tf.keras.layers.Input((256, 256, 3))
    print("Generating Architecture")
    model = create_unet(inputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model, show_shapes=True)
    print("Model's plot architecture done")
    callback = EarlyStopping(monitor="accuracy", patience=20, mode="auto")
    checkpoint = ModelCheckpoint(filepath="models/best_model.h5", save_best_only=True, save_weights_only=False,
                                 mode="auto", verbose=1, monitor="val_accuracy")
    current_day = date.today().strftime("%dd_%mm_%yyyy")
    tensorboard_cb = TensorBoard(log_dir=f"logs/cnn_model_{current_day}")

    model_history = model.fit(train_samples['images'], train_samples['masks'], epochs=200, verbose=1,
                              validation_split=0.3, callbacks=[callback, checkpoint, tensorboard_cb])

    plot_results(model_history, "accuracy", "conv_results.png")

    score = model.evaluate(test_samples['images'], test_samples['masks'])
    print("="*64)
    print("score", score)
    print("="*64)
    return model


if __name__ == '__main__':
    print("Reading Data")
    images, masks = read_dataset()
    print("Making Partitions")
    train_input_img, val_input_img, train_target_mask, val_target_mask = get_partitions(images, masks)
    train_samples_data = load_data(train_input_img, train_target_mask)
    test_samples_data = load_data(val_input_img, val_target_mask)
    model = create_and_compile_model(train_samples_data, test_samples_data)
    print("Predicting")
    predicted_masks, test_images, ground_truth_masks = predict_test_samples(test_samples_data, model)
    plot_images(test_images[20], predicted_masks[20], ground_truth_masks[20])
