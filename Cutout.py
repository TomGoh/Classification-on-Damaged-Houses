# import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

original_image = Image.open('archive\\test\\damage\\-93.6141_30.754263.jpeg')


# plt.imshow(original_image)
# plt.axis('off')
# plt.show()

def cutout(img, n_holes, length):
    global masked
    h = img.shape[0]
    w = img.shape[1]

    mask = np.ones((h, w), np.uint8)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0

        masked = cv2.bitwise_and(img, img, mask=mask)
    return masked

def cutout_preprocessing(image):
    return cutout(image,n_holes=10,length=8)
