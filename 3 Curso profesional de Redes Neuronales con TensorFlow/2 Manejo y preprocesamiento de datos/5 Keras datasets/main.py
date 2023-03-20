import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import json

with open("labels/cifar100_labels.json", "r") as fine_labels:
    cifar100_labels = json.load(fine_labels)

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
print(x_train.shape)
print(y_train.shape)


num_image = 40
y = y_train[num_image][0]
x = x_train[num_image]
y_name = cifar100_labels[y]
plt.imshow(x)
title = f"Class: {y_name} - Id Class: {y}"
plt.title(title)
plt.savefig("fig.png")
