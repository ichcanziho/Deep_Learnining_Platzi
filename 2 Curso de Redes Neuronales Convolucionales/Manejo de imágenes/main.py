import numpy as np
import matplotlib.pyplot as plt
from skimage import io  # Una alternativa podría ser opencv o pill

im = io.imread("imgs/perro.png")
print(im.shape)
print(im)


red_channel = im[:, :, 0]
green_channel = im[:, :, 1]
blue_channel = im[:, :, 2]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(red_channel, cmap="gray")
ax2.imshow(blue_channel, cmap="gray")
ax3.imshow(green_channel, cmap="gray")
ax1.set_title("Red channel")
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_title("Green channel")
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_title("Blue channel")
ax3.set_xticks([])
ax3.set_yticks([])
plt.savefig("imgs/channels.png")
plt.close()

aux_dim = np.zeros(im.shape[:2])
red = np.dstack((red_channel, aux_dim, aux_dim)).astype(np.uint8)
green = np.dstack((aux_dim, green_channel, aux_dim)).astype(np.uint8)
blue = np.dstack((aux_dim, aux_dim, blue_channel)).astype(np.uint8)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(red)
ax2.imshow(green)
ax3.imshow(blue)
ax1.set_title("Red")
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_title("Green")
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_title("Blue")
ax3.set_xticks([])
ax3.set_yticks([])
plt.savefig("imgs/channels2.png")
plt.close()


plt.imshow(im[270:530, 150:400])
plt.xticks([], [])
plt.yticks([], [])
plt.savefig("imgs/zoom.png")
plt.close()
