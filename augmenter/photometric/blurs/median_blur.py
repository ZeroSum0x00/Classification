import cv2
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image



def median_blur(image, ksize_norm=0.05):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))
        
    try:
        k_size = int(min(image.shape[:2]) * ksize_norm)
        k_size = k_size + 1 if k_size % 2 == 0 else k_size

        if k_size <= 2:
            return image
        return cv2.medianBlur(image, k_size)
    except:
        k_size = random.choice([3, 5, 7, 9, 13, 15, 17, 19])
        return cv2.medianBlur(image, k_size)


class MedianBlur(BaseTransform):
    def __init__(self, ksize_norm=0.05):
        self.ksize_norm = ksize_norm

    def image_transform(self, image):
        return median_blur(image, self.ksize_norm)


class RandomMedianBlur(BaseRandomTransform):
    def __init__(self, ksize_norm=0.8, prob=0.5):
        self.ksize_norm = ksize_norm
        self.prob       = prob

    @staticmethod
    def get_params(ksize):
        ksize_factor = 1.0

        if isinstance(ksize, numbers.Number) and ksize > 0:
            ksize_factor = random.uniform(0, ksize)
        else:
            if ksize[0] > 0 and ksize[1] > 0:  
                ksize_factor = random.uniform(ksize[0], ksize[1])

        if ksize_factor < 0:
            ksize_factor = 1.0

        return ksize_factor

    def image_transform(self, image):
        ksize_norm = self.get_params(self.ksize_norm)
        return median_blur(image, ksize_norm)
    