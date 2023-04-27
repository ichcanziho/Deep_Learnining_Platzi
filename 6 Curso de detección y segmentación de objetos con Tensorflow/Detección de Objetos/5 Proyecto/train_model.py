from utils import read_ini, preprocess_dataset, load_model, train_model
import pandas as pd


if __name__ == '__main__':
    model_dir, config_dir, labelmap_dir, dataset_dir = read_ini()
    detection_model = load_model(model_dir, config_dir, n_classes=1)
    print("Model Has Been Loaded Successfully")

    dataset = pd.read_csv("data/dataset.csv")

    n_classes = 1
    np_images, np_boxes, np_labels = preprocess_dataset(dataset, dataset_dir)
    train_model(detection_model, np_images, np_boxes, np_labels, n_classes,
                batch_size=10, learning_rate=0.01, num_batches=20)
