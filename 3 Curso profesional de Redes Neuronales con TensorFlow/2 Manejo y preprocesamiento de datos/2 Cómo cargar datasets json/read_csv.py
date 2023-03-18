import pandas as pd
import matplotlib.pyplot as plt


def plot_img(img, label):
    plt.imshow(img, cmap="gray")
    plt.title(f"label = {label}")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("test_csv.png")


if __name__ == '__main__':

    data = pd.read_csv("../datasets/sign_mnist_train/sign_mnist_train.csv")
    samples = len(data)
    print("samples:", samples)
    print(data)
    y = data["label"].values
    X = data.drop('label', axis=1).values.reshape((samples, 28, 28))
    for img, label in zip(X, y):
        plot_img(img, label)
        break

