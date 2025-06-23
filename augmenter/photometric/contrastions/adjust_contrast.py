import cv2
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image



def adjust_contrast(image, contrast_factor):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))

    img = image.astype(np.float32)
    mean = round(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean())
    img = (1 - contrast_factor) * mean + contrast_factor * img
    img = img.clip(min=0, max=255)
    return img.astype(image.dtype)


class AdjustContrast(BaseTransform):
    def __init__(self, contrast_factor=1):
        self.contrast_factor = contrast_factor

    def image_transform(self, image):
        return adjust_contrast(image, self.contrast_factor)


class RandomAdjustContrast(BaseRandomTransform):
    def __init__(self, contrast_range=1, prob=0.5):
        assert (isinstance(contrast_range, (int, float)) and contrast_range > 0) or (isinstance(contrast_range, collections.Iterable) and len(contrast_range) == 2)
        self.contrast_range = contrast_range
        self.prob = prob

    @staticmethod
    def get_params(contrast):
        contrast_factor = 1.0

        if isinstance(contrast, numbers.Number):
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        elif isinstance(contrast, (tuple, list)):
            if contrast[0] > 0 and contrast[1] > 0:
                contrast_factor = random.uniform(contrast[0], contrast[1])

        if contrast_factor < 0:
            contrast_factor = 1.0

        return contrast_factor

    def image_transform(self, image):
        contrast_factor = self.get_params(self.contrast_range)
        return adjust_contrast(image, contrast_factor)
    