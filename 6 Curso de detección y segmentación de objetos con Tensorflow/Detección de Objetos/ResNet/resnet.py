import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_image_classified(image, bboxes, c_index):
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

    plt.figure(figsize=(12, 16))
    plt.imshow(image_np_with_detections)
    plt.tight_layout()
    plt.savefig("anotacion.png")
    plt.close()


def get_model_detection_function(model):
    @tf.function
    def detect_fn_(image):
        image, shape = model.preprocess(image)
        prediction_dict = model.predict(image, shape)
        detections_ = model.postprocess(prediction_dict, shape)

        return detections_

    return detect_fn_


def load_model():

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/" \
           "modelo"
    model_name = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'

    pipeline_config = os.path.join(root + '/models/research/object_detection/configs/tf2/' + model_name + '.config')
    model_dir = f"{root}/{model_name}/checkpoint/"
    label_map_path = f'{root}/{model_name}/mscoco_label_map.pbtxt'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    detect_fn = get_model_detection_function(detection_model)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index_ = label_map_util.create_category_index(categories)

    return detect_fn, category_index_


def image_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)


if __name__ == '__main__':
    img = cv2.imread("1479506176491553178.jpg")
    image_np, input_tensor = image_to_tensor(img)
    print("Loaded image")
    predict, category_index = load_model()
    print("Loaded Model")
    boxes = predict(input_tensor)
    print("BBoxes generated")
    show_image_classified(image_np, boxes, category_index)
    print("Image saved")
