import xml.etree.ElementTree as ET
import pandas as pd
import glob
import os


def generate_dataframe(dataset_dir, name2id):

    dataset = {"filename": [], "width": [], "height": [], "xmin": [], "ymin": [], "xmax": [], "ymax": [], "class": [],
               "class_id": []}

    for item in glob.glob(os.path.join(dataset_dir, "*.xml")):
        tree = ET.parse(item)
        filename, width, height, xmin, ymin, xmax, ymax, name = None, None, None, None, None, None, None, None
        for elem in tree.iter():
            if 'filename' in elem.tag:
                filename = elem.text
            elif 'width' in elem.tag:
                width = int(elem.text)
            elif 'height' in elem.tag:
                height = int(elem.text)
            elif 'name' in elem.tag:
                name = elem.text
            elif 'xmin' in elem.tag:
                xmin = int(elem.text)
            elif 'ymin' in elem.tag:
                ymin = int(elem.text)
            elif 'xmax' in elem.tag:
                xmax = int(elem.text)
            elif 'ymax' in elem.tag:
                ymax = int(elem.text)

                dataset['filename'].append(filename)
                dataset['width'].append(width)
                dataset['height'].append(height)
                dataset['xmin'].append(xmin)
                dataset['ymin'].append(ymin)
                dataset['xmax'].append(xmax)
                dataset['ymax'].append(ymax)
                dataset["class"].append(name)
                dataset["class_id"].append(name2id[name])
    df = pd.DataFrame(dataset)
    return df


if __name__ == '__main__':
    labelmap = {"d1": 1, "d2": 2, "d3": 3, "d4": 4, "d5": 5}
    root = "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/project/dataset"
    paths = {"train_labels": f"{root}/train_labels",
             "test_labels": f"{root}/test_labels",
             "train": f"{root}/train",
             "test": f"{root}/test"}

    train = generate_dataframe(paths["train_labels"], labelmap)
    print(train)
    train.to_csv(f"{paths['train']}/train_labels.csv", index=False)
    test = generate_dataframe(paths["test_labels"], labelmap)
    print(test)
    test.to_csv(f"{paths['test']}/train_labels.csv", index=False)
