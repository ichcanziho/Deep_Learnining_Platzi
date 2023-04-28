from sklearn.model_selection import train_test_split
# from config import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os

paths = {"dataset": "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/Segmentation/dataset"}


def predict_test_samples(val_map, model_):
    img = val_map['images']
    mask = val_map['masks']

    test_images_ = np.array(img)

    predictions = model_.predict(test_images_)

    return predictions, test_images_, mask


def plot_images(test_image, predicted_maks, ground_truth):
    plt.rcParams["figure.figsize"] = [10, 3.50]
    plt.rcParams["figure.autolayout"] = True

    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.title('Image')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_maks)
    plt.title('Predicted mask')

    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth)
    plt.title('Ground truth mask')
    plt.savefig("predictions.png")

def read_dataset():
    dataset_path = paths["dataset"]
    filenames = glob.glob(os.path.join(dataset_path, "*.png"))
    filenames.sort()
    image_list = []
    masks_list = []
    for filename in filenames:
        if filename.endswith("L.png"):
            masks_list.append(filename)
        else:
            image_list.append(filename)
    print(len(image_list), len(masks_list))
    return image_list, masks_list


def get_partitions(image_list, masks_list, ts=0.2):
    return train_test_split(image_list, masks_list, test_size=ts, random_state=42)


def load_data(images_path, masks_path):
    samples = {'images': [], 'masks': []}

    for i in range(len(images_path)):
        img = plt.imread(images_path[i])
        mask = plt.imread(masks_path[i])
        img = cv2.resize(img, (256, 256))
        masks = cv2.resize(mask, (256, 256))

        samples['images'].append(img)
        samples['masks'].append(masks)

    samples = {
        'images': np.array(samples['images']),
        'masks': np.array(samples['masks']),
    }
    return samples


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




if __name__ == '__main__':
    images, masks = read_dataset()
    train_samples = load_data(images, masks)
    plt.rcParams["figure.figsize"] = [8, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.subplot(1, 2, 1)
    plt.imshow(train_samples['images'][10])
    plt.subplot(1, 2, 2)
    plt.imshow(train_samples['masks'][10])
    plt.savefig("example.png")
