import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
from config import paths
import tensorflow as tf
import numpy as np
import cv2


def get_model_detection_function(model):
    @tf.function
    def detect_fn_(image):
        image, shape = model.preprocess(image)
        prediction_dict = model.predict(image, shape)
        detections_ = model.postprocess(prediction_dict, shape)

        return detections_

    return detect_fn_


def load_model():
    category_index = label_map_util.create_category_index_from_labelmap(paths['LABELMAP'])
    configs = config_util.get_configs_from_pipeline_file(paths['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths["CKPT"], 'ckpt-0')).expect_partial()
    detect_fn = get_model_detection_function(detection_model)
    return detect_fn, category_index


def show_bboxes(image, bboxes, c_index):
    detections = bboxes
    label_id_offset = 1
    image_np_with_detections = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        c_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.7
    )
    return image_np_with_detections


def test_single_image(image):
    img = cv2.imread(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    print("Loaded image")
    detect, labels = load_model()
    print("loaded model")
    boxes = detect(input_tensor)
    print("BBoxes generated")
    annotated = show_bboxes(img, boxes, labels)
    cv2.imwrite("annotated.png", annotated)
    print("Image saved")


if __name__ == '__main__':
    im = "coche.jpg"
    test_single_image(im)
