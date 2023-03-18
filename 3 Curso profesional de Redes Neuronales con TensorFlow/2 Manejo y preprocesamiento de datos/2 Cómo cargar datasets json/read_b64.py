from json import load
import base64
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def b64_to_np(b_string: str):
    jpg_original = base64.b64decode(b_string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image_buffer = cv.imdecode(jpg_as_np, flags=1)
    return image_buffer


def plot_img(img, label):
    plt.imshow(img, cmap="gray")
    plt.title(f"label = {label}")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("test_b64.png")


if __name__ == '__main__':

    with open("../datasets/sign_mnist_base64/data2.json", "r", encoding="utf-8") as d:
        data = load(d)

    X, y = [], []
    for example in data:
        for label, b_image in example.items():
            print(label, "-", b_image)
            img = b64_to_np(b_image)
            X.append(img)
            y.append(label)
            plot_img(img, label)
        break
