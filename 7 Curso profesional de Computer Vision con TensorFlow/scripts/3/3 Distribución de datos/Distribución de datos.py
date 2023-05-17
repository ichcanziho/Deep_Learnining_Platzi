import os
import random
import shutil


def split_data(paths, train_ratio):
    train = train_ratio
    # Obtenemos una lista con todas las imágenes de la ruta `dataset`
    content = os.listdir(paths["dataset"])
    for nCount in range(int(len(content) * train)):
        # Seleccionamos una imagen aleatoria
        random_choice_img = random.choice(content)
        random_choice_img_abs = "{}/{}".format(paths["dataset"], random_choice_img)
        target_img = "{}/{}".format(paths["train"], random_choice_img)
        # Copiamos la imagen de la ruta actual a la ruta objetivo
        shutil.copyfile(random_choice_img_abs, target_img)
        # Eliminamos de la lista de posibles opciones, la imagen actual que hemos seleccionado. Esto con el fin de
        # evitar imágenes repetidas entre los conjuntos de training y testing.
        content.remove(random_choice_img)

    # La lista de imágenes que queda en `content` representan el testing set
    for img in content:
        random_choice_img_abs = "{}/{}".format(paths["dataset"], img)
        target_img = "{}/{}".format(paths["test"], img)
        # Aquí solo es necesario copiar las imágenes a la nueva ruta
        shutil.copyfile(random_choice_img_abs, target_img)


def make_dirs(paths):
    for _, path in paths.items():
        if not os.path.exists(path):
            os.mkdir(path)


if __name__ == '__main__':
    dirs = {"root": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/",
            "dataset": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/dataset",
            "clean": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean",
            "train": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean/train",
            "test": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean/test"}

    make_dirs(dirs)

    split_data(dirs, train_ratio=0.7)
