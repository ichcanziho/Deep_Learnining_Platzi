import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from keras.models import load_model
from core.utils import predict_test_samples, plot_images, load_data


if __name__ == '__main__':

    test_sample = load_data(["test/0016E5_08159.png"], ["test/0016E5_08159_L.png"])
    model = load_model("core/models/best_model.h5")
    predicted_masks, test_images, ground_truth_masks = predict_test_samples(test_sample, model)
    plot_images(test_images[0], predicted_masks[0], ground_truth_masks[0])
