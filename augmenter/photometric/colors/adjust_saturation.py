import cv2
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def adjust_saturation(image, saturation_factor):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    img = image.astype(np.float32)
    degenerate = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    img = (1 - saturation_factor) * degenerate + saturation_factor * img
    img = img.clip(min=0, max=255)
    return img.astype(image.dtype)


class AdjustSaturation(BaseTransform):
    def __init__(self, saturation_factor=1):
        self.saturation_factor = saturation_factor

    def image_transform(self, image):
        return adjust_saturation(image, self.saturation_factor)


class RandomAdjustSaturation(BaseRandomTransform):
    def __init__(self, saturation_range=1, prob=0.5):
        assert (isinstance(saturation_range, (int, float)) and saturation_range > 0) or (isinstance(saturation_range, collections.Iterable) and len(saturation_range) == 2)
        self.saturation_range = saturation_range
        self.prob = prob

    @staticmethod
    def get_params(saturation):
        saturation_factor = 1.0
        
        if isinstance(saturation, numbers.Number) and saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        elif isinstance(saturation, (tuple, list)):
            if saturation[0] > 0 and saturation[1] > 0:
                saturation_factor = random.uniform(saturation[0], saturation[1])
        
        if saturation_factor < 0:
            saturation_factor = 1.0

        return saturation_factor

    def image_transform(self, image):
        saturation_factor = self.get_params(self.saturation_range)
        return adjust_saturation(image, saturation_factor)