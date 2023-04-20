from object_detection.utils import visualization_utils as viz_utils
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import configparser
import numpy as np
import glob
import cv2
import os


def read_ini():
    config = configparser.ConfigParser()
    config.read("config.ini")
    out = {}
    for section in config.sections():
        for key in config[section]:
            out[key] = config[section][key]

    return out["model"], out["config"], out["labelmap"], out["dataset"]


def generate_dataframe():
    _, _, _, dataset_dir = read_ini()

    dataset = {"file": [], "width": [], "height": [], "xmin": [], "xmax": [], "ymin": [], "ymax": []}

    for item in glob.glob(os.path.join(dataset_dir+"/annotations", "*.xml")):
        tree = ET.parse(item)
        for elem in tree.iter():
            if 'filename' in elem.tag:
                filename = elem.text
            elif 'width' in elem.tag:
                width = int(elem.text)
            elif 'height' in elem.tag:
                height = int(elem.text)
            elif 'xmin' in elem.tag:
                xmin = int(elem.text)
            elif 'ymin' in elem.tag:
                ymin = int(elem.text)
            elif 'xmax' in elem.tag:
                xmax = int(elem.text)
            elif 'ymax' in elem.tag:
                ymax = int(elem.text)

                dataset['file'].append(filename)
                dataset['width'].append(width)
                dataset['height'].append(height)
                dataset['xmin'].append(xmin/width)
                dataset['ymin'].append(ymin/height)
                dataset['xmax'].append(xmax/width)
                dataset['ymax'].append(ymax/height)

    df = pd.DataFrame(dataset)
    df["label_id"] = 1
    df.to_csv("data/dataset.csv", index=False)
    print(df)


def plot_detections(im_array, boxes, classes, scores, category_index):
    aux = im_array.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(aux, boxes, classes, scores, category_index,
                                                        use_normalized_coordinates=True)
    return aux


def plot_example(boxes: dict, limit: int, layout: tuple, dataset_dir: str) -> None:
    labelmap = {1: {'id': 1, 'name': 'plate'}}
    limit_images = limit
    i_image = 0
    plt.figure(figsize=(30, 30))
    for key, value in boxes.items():
        bboxes = value["boxes"]
        classes = value["ids"]
        im = cv2.imread(dataset_dir + "/images/" + key)
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


def get_bounding_boxes(data):
    gt_boxes = {}
    for index, row in data.iterrows():
        id_label = row["label_id"]
        bbox = np.array([[row['ymin'], row['xmin'], row['ymax'], row['xmax']]], dtype=np.float32)
        im_name = row['file']
        if im_name not in gt_boxes:
            gt_boxes[im_name] = {"boxes": np.array(bbox),
                                 "ids": np.array([id_label])}
        else:
            gt_boxes[im_name] = {"boxes": np.append(gt_boxes[im_name]["boxes"], np.array(bbox), axis=0),
                                 "ids": np.append(gt_boxes[im_name]["ids"], np.array([id_label]), axis=0)}

    return gt_boxes


if __name__ == '__main__':
    generate_dataframe()
