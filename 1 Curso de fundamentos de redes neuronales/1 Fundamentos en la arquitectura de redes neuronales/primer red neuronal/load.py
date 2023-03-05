import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical

model = load_model("Model/numeros.h5")
(_, _), (test_data, test_labels) = mnist.load_data()

x_test = test_data.reshape((10000, 28*28))
x_test = x_test.astype("float32")/255

y_test = to_categorical(test_labels)

model.evaluate(x_test, y_test)
