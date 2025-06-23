import cv2
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image



def erosion_or_dilation(image, kernel_size=5, reversed=False):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))

    img = image.copy()
    kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    er, dil = cv2.erode, cv2.dilate
    
    if reversed:
        er, dil = dil, er

    img[:, :, 0] = er(img[:, :, 0], kernel, iterations=1)
    img[:, :, 1] = er(img[:, :, 1], kernel, iterations=1)
    img[:, :, 2] = er(img[:, :, 2], kernel, iterations=1)

    if img.shape[2] > 3:
        img[:, :, 3] = dil(img[:, :, 3], kernel, iterations=1)

    return img


class Erosion(BaseTransform):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def image_transform(self, image):
        return erosion_or_dilation(image, self.kernel_size, reversed=False)


class RandomErosion(BaseRandomTransform):
    def __init__(self, kernel_size=5, prob=0.5):
        self.kernel_size = kernel_size
        self.prob = prob

    def image_transform(self, image):
        return erosion_or_dilation(image, self.kernel_size, reversed=False)


class Dilation(BaseTransform):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def image_transform(self, image):
        return erosion_or_dilation(image, self.kernel_size, reversed=True)


class RandomDilation(BaseRandomTransform):
    def __init__(self, kernel_size=5, prob=0.5):
        self.kernel_size = kernel_size
        self.prob = prob

    def image_transform(self, image):
        return erosion_or_dilation(image, self.kernel_size, reversed=True)
    