import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np


def load_model(model, labels):
    detect_fn = tf.saved_model.load(model)
    # model = tf.saved_model.load(model)
    # detect_fn = model.signatures['serving_default']
    category_index = label_map_util.create_category_index_from_labelmap(labels)
    return detect_fn, category_index


def classify(image, detect_function, labels):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

    detections = detect_function(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        labels,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        use_normalized_coordinates=True
    )

    return image_np_with_detections, detections

