import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import read_ini

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from object_detection.utils import visualization_utils as viz_utils
import numpy as np


def get_bounding_boxes(mask: tuple, width: int, height: int) -> dict:
    """
    Returns a dictionary with the correspondent bounding boxes
    :param mask: tuple specifying the `class_id` you want to keep. Example: (1, 2, 3, 4, 5)
    :param width: Image width
    :param height: Image Height
    :return: gt_boxes
    """
    gt_boxes = {}
    for index, row in data.iterrows():
        id_label = row["class_id"]
        if id_label in mask:
            bbox = np.array([[row['ymin'] / height, row['xmin'] / width, row['ymax'] / height, row['xmax'] / width]],
                            dtype=np.float32)

            im_name = row['frame']
            if im_name not in gt_boxes:
                gt_boxes[im_name] = {"boxes": np.array(bbox),
                                     "ids": np.array([id_label])}
            else:
                gt_boxes[im_name] = {"boxes": np.append(gt_boxes[im_name]["boxes"], np.array(bbox), axis=0),
                                     "ids": np.append(gt_boxes[im_name]["ids"], np.array([id_label]), axis=0)}

    return gt_boxes


def plot_detections(im_array, boxes, classes, scores, category_index):
    aux = im_array.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(aux, boxes, classes, scores, category_index,
                                                        use_normalized_coordinates=True)
    return aux


def plot_example(boxes: dict, limit: int, layout: tuple) -> None:
    """
    Makes an image with multiple detection object examples
    :param boxes: Dictionary with the image name as key and bounding boxes and label ids as a value
    :param limit: Number of examples to include
    :param layout: Distribution of the examples. Example given a limit of 4 a valid layout could be (1, 4), (2, 2) (4, 1)
    :return: None
    """
    limit_images = limit
    i_image = 0
    plt.figure(figsize=(30, 30))
    for key, value in boxes.items():
        bboxes = value["boxes"]
        classes = value["ids"]
        im = cv2.imread(root + "/images/" + key)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dummy_scores = np.ones(shape=[bboxes.shape[0]], dtype=np.float32)
        a = plot_detections(im, bboxes, classes, dummy_scores, labelmap)
        plt.subplot(layout[0], layout[1], i_image + 1)
        plt.imshow(a)
        if i_image >= limit_images-1:
            break
        i_image += 1

    plt.savefig("object_detection.png")
    plt.close()


if __name__ == '__main__':
    root = read_ini()["dataset"]

    img = cv2.imread(f'{root}/images/1479506176491553178.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.savefig("ejemplo.png")
    plt.close()
    print("img shape:", img.shape)

    data = pd.read_csv(f"{root}/labels_train.csv")
    print(data)
    print("train images", len(data["frame"].unique()))
    labelmap = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'truck'}, 3: {'id': 3, 'name': 'pedestrian'},
                4: {'id': 4, 'name': 'bicyclist'}, 5: {'id': 5, 'name': 'light'}}

    im_width, im_height = img.shape[1], img.shape[0]
    get_boxes = get_bounding_boxes(mask=(1, 3), width=im_width, height=im_height)
    plot_example(boxes=get_boxes, limit=20, layout=(5, 4))
