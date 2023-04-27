from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf
import pandas as pd
import configparser
import numpy as np
import random
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

    for item in glob.glob(os.path.join(dataset_dir + "/annotations", "*.xml")):
        tree = ET.parse(item)
        filename, width, height, xmin, ymin, xmax, ymax = None, None, None, None, None, None, None
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
                dataset['xmin'].append(xmin / width)
                dataset['ymin'].append(ymin / height)
                dataset['xmax'].append(xmax / width)
                dataset['ymax'].append(ymax / height)

    df = pd.DataFrame(dataset)
    df["label_id"] = 1
    counts = df.groupby('file').size()
    df = df[df['file'].isin(counts[counts == 1].index)]
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
        if i_image >= limit_images - 1:
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


def load_model(model_dir, config_dir, n_classes):
    configs = config_util.get_configs_from_pipeline_file(config_dir)
    model_config = configs['model']
    model_config.ssd.num_classes = n_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(model_config=model_config, is_training=True)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        _box_prediction_head=detection_model._box_predictor._box_prediction_head
    )
    fake_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model.feature_extractor,
        _box_predictor=fake_box_predictor
    )
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)

    return detection_model


def transform_boxes(b):
    return [b[0], b[2], b[1], b[3]]


def preprocess_data_example(image_dir, boxes, label):

    im = cv2.imread(image_dir)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    boxes = transform_boxes(boxes)
    boxes.append(label)
    return im, boxes


def preprocess_dataset(dataset, dataset_dir):

    np_images, np_boxes, np_labels = [], [], []
    for _, row in dataset.iterrows():
        file = row[0]
        file = dataset_dir + "/images/" + file
        bbox = row[3:7].squeeze().values
        lb = row[-1]
        np_image, np_box = preprocess_data_example(file, bbox, lb)
        np_images.append(np_image)
        np_boxes.append(np_box)
        np_labels.append(lb)

    return np_images, np_boxes, np_labels


def get_model_train_step_function(model, optimizer, vars_to_fine_tune, batch_size):
    @tf.function
    def train_step_fn(image_tensors, ground_truth_boxes_list, ground_truth_classes_list):
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(groundtruth_boxes_list=ground_truth_boxes_list,
                                  groundtruth_classes_list=ground_truth_classes_list)

        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat([model.preprocess(image_tensor)[0]
                                             for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss
    return train_step_fn


def process_data_augmentation(images, bounding_boxes, labels, num_classes, transform):
    image_tensors = []
    boxes_tensors = []
    label_tensors = []
    offset = 1
    for im, box, label in zip(images, bounding_boxes, labels):
        transformed = transform(image=im, bboxes=[box])
        transformed_image = transformed['image']
        boxes = transformed['bboxes'][0]
        transformed_bbox = transform_boxes(boxes)
        image_tensor = tf.expand_dims(tf.convert_to_tensor(transformed_image, dtype=tf.float32), axis=0)
        np_box_ = np.array([transformed_bbox])
        bbox_tensor = tf.convert_to_tensor(np_box_, dtype=tf.float32)
        v = [label for _ in range(np_box_.shape[0])]
        v = np.array(v, dtype=np.int32) - offset
        zero_indexed_ground_truth_classes = tf.convert_to_tensor(v)
        label_tensor = tf.one_hot(zero_indexed_ground_truth_classes, num_classes)
        image_tensors.append(image_tensor)
        boxes_tensors.append(bbox_tensor)
        label_tensors.append(label_tensor)

    return image_tensors, boxes_tensors, label_tensors


def detect(image_np, detection_model):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    predict_dict = detection_model.predict(preprocessed_image, shapes)
    return detection_model.postprocess(predict_dict, shapes)


def train_model(model, np_images, np_boxes, np_labels, n_classes, batch_size=10, learning_rate=0.01, num_batches=20):

    transform = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.8)],
                          bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3))

    tf.keras.backend.set_learning_phase(True)
    trainable_variables = model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']

    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)
    print("Setting up trainable variables")
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    print("Optimizer Initialized")
    train_step_fn = get_model_train_step_function(model, optimizer, to_fine_tune, batch_size)
    print("Train step function created")
    print("Training is starting")
    for idx in range(num_batches):
        all_keys = list(range(len(np_images)))
        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]

        np_images_subset = [np_images[key] for key in example_keys]
        np_boxes_subset = [np_boxes[key] for key in example_keys]
        np_labels_subset = [np_labels[key] for key in example_keys]

        tensor_images, tensor_boxes, tensor_labels = process_data_augmentation(np_images_subset, np_boxes_subset,
                                                                               np_labels_subset, n_classes, transform)

        total_loss = train_step_fn(tensor_images, tensor_boxes, tensor_labels)

        if idx % 10 == 0:
            print('batch ' + str(idx) + ' of ' + str(num_batches) + ', loss= ' + str(total_loss), flush=True)

    print("Training finished")

    category_index = {1: {'id': 1, 'name': 'plate'}}
    img = cv2.imread('Cars0.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detect(img, model)

    label_id_offset = 1
    image_np_with_detections = img.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.7
    )

    plt.figure(figsize=(12, 16))
    plt.imshow(image_np_with_detections)
    plt.savefig("detections.png")
    plt.close()

    # model.build((640, 640, 3))
    # tf.saved_model.save(model, 'plate_model', signatures=None, options=None)


if __name__ == '__main__':
    generate_dataframe()
