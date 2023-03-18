import requests
from json import loads
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def plot_img(img, label):
    plt.imshow(img, cmap="gray")
    plt.title(f"label = {label}")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("test_json.png")


if __name__ == '__main__':

    with open("../datasets/sign_mnist_json/data.json", "r", encoding='utf-8') as d:
        data = [loads(line) for line in d.readlines()]

    X, y = [], []
    for example in data:
        print(example)
        url_image = example.get("content", 0)
        label = example.get("label", 0)
        # Petición al servidor
        response = requests.get(url_image).content
        print(type(response), response)
        # transformado `bytes` en PIL image
        pil_image = Image.open(BytesIO(response))
        print(pil_image)
        # transformando pil_image en un numpy array
        img = np.asarray(pil_image).reshape(28, 28)
        X.append(img)
        y.append(label)
        plot_img(img, label)
        break
