import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
import albumentations as A


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


def transform_boxes(bboxes):
    return np.array([[b[1], b[0], b[3], b[2]] for b in bboxes])


def apply_augmentation(img: np.array, bboxes: dict, q: int) -> dict:
    """
    Make q new images using a transform object
    :param img: original image to be transformed
    :param bboxes: original bounding boxes including classes id
    :param q: number of image to create from the original one
    :return: dict
    """

    ab_box = transform_boxes(bboxes["boxes"])
    label = bboxes["ids"]
    ab_box = [list(row) + [label[i]] for i, row in enumerate(ab_box)]
    print(ab_box)
    images_, boxes_, labels_ = [], [], []
    for _ in range(q):
        transformed = transform(image=img, bboxes=ab_box)
        transformed_image = transformed['image']
        transformed_bboxes = transform_boxes(transformed['bboxes'])
        images_.append(transformed_image)
        boxes_.append(transformed_bboxes)
        labels_.append(label)
    return {"images": images_, "bboxes": boxes_, "labels": labels_}


def plot_detections(im_array, boxes, classes, scores, category_index):
    aux = im_array.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(aux, boxes, classes, scores, category_index,
                                                        use_normalized_coordinates=True)
    return aux


def plot_example(im_to_plot: dict, layout: tuple) -> None:
    """
    Makes an image with multiple detection object examples
    :param im_to_plot: dictionary containing images, bboxes and labels.
    :param layout: Distribution of the examples. Example given a limit of 4 a valid layout could be (1, 4), (2, 2) (4, 1)
    :return: None
    """
    plt.figure(figsize=(30, 30))
    images = im_to_plot["images"]
    bboxes = im_to_plot["bboxes"]
    labels = im_to_plot["labels"]

    for index in range(len(images)):
        img = images[index]
        bbox = bboxes[index]
        label = labels[index]
        dummy_scores = np.ones(shape=[bbox.shape[0]], dtype=np.float32)
        a = plot_detections(img, bbox, label, dummy_scores, labelmap)
        plt.subplot(layout[0], layout[1], index + 1)
        plt.imshow(a)

    plt.tight_layout()
    plt.savefig("object_detection2.png")
    plt.close()


if __name__ == '__main__':

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/dataset"

    data = pd.read_csv(f"{root}/labels_train.csv", nrows=2)
    get_boxes = get_bounding_boxes(mask=(1, 3), width=480, height=300)
    labelmap = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'truck'}, 3: {'id': 3, 'name': 'pedestrian'},
                4: {'id': 4, 'name': 'bicyclist'}, 5: {'id': 5, 'name': 'light'}}

    print(data)
    print(get_boxes)
    transform = A.Compose([A.HorizontalFlip(p=0.5),
                           A.RandomBrightnessContrast(p=0.8),
                           A.ShiftScaleRotate(scale_limit=0.9, rotate_limit=10, p=0.7)],
                          bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3))

    for key, value in get_boxes.items():
        image = cv2.imread(root + "/images/" + key)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ans = apply_augmentation(image, value, 3)
        plot_example(ans, layout=(3, 1))
