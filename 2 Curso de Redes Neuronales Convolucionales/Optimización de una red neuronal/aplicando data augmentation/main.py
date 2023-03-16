import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

if __name__ == '__main__':
    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest',
                                 brightness_range=[0.4, 1.5]
                                 )

    img = load_img('imgs/perro.png')
    x = img_to_array(img)
    print(x.shape)
    x = x.reshape((1,) + x.shape)
    print(x.shape)

    for i, batch in enumerate(datagen.flow(x, batch_size=1)):
        plt.imshow(array_to_img(batch[0]))
        plt.savefig(f"imgs/transformation_{i}.png")
        plt.close()
        if i == 3:
            break

    """
    train_generator = datagen.flow_from_directory(
    '/train',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
    )
    
    train/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    
    """