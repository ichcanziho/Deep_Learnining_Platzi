import imageio
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1] + 1, step):
        for x in range(0, image.shape[1] - ws[0] + 1, step):
            yield x, y, image[y:y + ws[1], x:x + ws[0]]


def get_window(window):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.imshow(img)
    rect = patches.Rectangle((window[0], window[1]), 200, 200, linewidth=2, edgecolor='g', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(window[2])
    ax2.set_xticks([])
    ax2.set_yticks([])
    # fig.tight_layout()
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
    img_array = img_array[380:840, 140:1090]
    plt.close()
    return img_array


if __name__ == '__main__':
    img = cv2.imread("mujer.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(1000, 1000))

    img_gif = []
    for w in sliding_window(img, 200, (200, 200)):
        img_gif.append(get_window(w))

    imageio.mimwrite('animación.gif', img_gif, 'GIF', duration=0.2)

