import cv2
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def motion_blur(image, ksize_norm=0.1):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))
        
    k_size = int(min(image.shape[:2]) * ksize_norm)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size
    if k_size <= 2:
        return image
    
    x1, x2 = random.randint(0, k_size - 1), random.randint(0, k_size - 1)
    y1, y2 = random.randint(0, k_size - 1), random.randint(0, k_size - 1)

    kernel_mtx = np.zeros((k_size, k_size), dtype=np.uint8)
    cv2.line(kernel_mtx, (x1, y1), (x2, y2), 1, thickness=1)
    return cv2.filter2D(image, -1, kernel_mtx / np.sum(kernel_mtx))


class MotionBlur(BaseTransform):
    def __init__(self, ksize_norm=0.1):
        self.ksize_norm = ksize_norm

    def image_transform(self, image):
        return motion_blur(image, self.ksize_norm)


class RandomMotionBlur(BaseRandomTransform):
    def __init__(self, ksize_norm=0.1, prob=0.5):
        self.ksize_norm = ksize_norm
        self.prob       = prob

    def image_transform(self, image):
        return motion_blur(image, self.ksize_norm)
 