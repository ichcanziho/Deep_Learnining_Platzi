from core.predict import load_model, classify
from config import paths
import cv2
import numpy as np
from PIL import Image


def test_single_image(image):
    print("starting")
    image = cv2.imread(image)
    # image = np.array(Image.open(image))

    detection, labels = load_model(**paths)
    image_o, bboxes = classify(image, detection, labels)
    cv2.imwrite("annotated2.png", image_o)
    print("finish")


if __name__ == '__main__':
    im = "coche.jpg"
    test_single_image(im)
