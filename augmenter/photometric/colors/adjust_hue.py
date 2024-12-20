import cv2
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def adjust_hue(image, hue_factor):
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('Hue factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    img = image.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(hue_factor * 255)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return img.astype(image.dtype)


class AdjustHue(BaseTransform):
    def __init__(self, hue_factor=1):
        self.hue_factor = hue_factor

    def image_transform(self, image):
        return adjust_hue(image, self.hue_factor)


class RandomAdjustHue(BaseRandomTransform):
    def __init__(self, hue_range=1, prob=0.5):
        assert (isinstance(hue_range, (int, float)) and hue_range > 0) or (isinstance(hue_range, collections.Iterable) and len(hue_range) == 2)
        self.hue_range = hue_range
        self.prob      = prob

    @staticmethod
    def get_params(hue):
        hue_factor = 0.0

        if isinstance(hue, numbers.Number) and hue > 0:
            hue_factor = random.uniform(-hue, hue)
        elif isinstance(hue, (tuple, list)):
            hue_factor = random.uniform(hue[0], hue[1])

        if not(-0.5 <= hue_factor <= 0.5):
            hue_factor = 0.0
            
        return hue_factor

    def image_transform(self, image):
        hue_factor = self.get_params(self.hue_range)
        return adjust_hue(image, hue_factor)