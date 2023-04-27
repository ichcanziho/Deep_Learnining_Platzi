from utils import read_ini, get_bounding_boxes, plot_example

import pandas as pd


if __name__ == '__main__':
    model_dir, config_dir, labelmap_dir, dataset_dir = read_ini()
    data = pd.read_csv("data/dataset.csv")
    bboxes = get_bounding_boxes(data)
    plot_example(bboxes, 6, (2, 3), dataset_dir)
