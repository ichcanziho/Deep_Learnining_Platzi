import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage import io, color

img = io.imread('input/gabriel.png')
print("Original's shape:", img.shape)
img_gray = color.rgb2gray(img)
print("Gray's shape:", img_gray.shape)

# Kernel para detectar bordes verticales

kernel_v = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])

# Kernel para detectar bordes horizontales

kernel_h = np.array([[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]])

img_kernel_v = nd.convolve(img_gray, kernel_v)
img_kernel_h = nd.convolve(img_gray, kernel_h)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
ax1.imshow(img)
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(img_gray, cmap="gray")
ax2.set_title('Gray scale')
ax2.axis('off')

ax3.imshow(img_kernel_v, cmap="gray")
ax3.set_title('Vertical Kernel Conv')
ax3.axis('off')

ax4.imshow(img_kernel_h, cmap="gray")
ax4.set_title('Horizontal Kernel Conv')
ax4.axis('off')


plt.savefig("imgs/resultados.png")
