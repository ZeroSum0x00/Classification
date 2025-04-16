import cv2
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def adjust_brightness(image, brightness_factor):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))
        
    img = image.astype(np.float32) * brightness_factor
    img = img.clip(min=0, max=255)
    return img.astype(image.dtype)


class AdjustBrightness(BaseTransform):
    def __init__(self, brightness_factor=1):
        self.brightness_factor = brightness_factor

    def image_transform(self, image):
        return adjust_brightness(image, self.brightness_factor)


class RandomAdjustBrightness(BaseRandomTransform):
    def __init__(self, brightness_range=1, prob=0.5):
        assert (isinstance(brightness_range, (int, float)) and brightness_range > 0) or (isinstance(brightness_range, collections.Iterable) and len(brightness_range) == 2)
        self.brightness_range = brightness_range
        self.prob = prob

    @staticmethod
    def get_params(brightness):
        brightness_factor = 1.0

        if isinstance(brightness, numbers.Number):
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            if brightness[0] > 0 and brightness[1] > 0:
                brightness_factor = random.uniform(brightness[0], brightness[1])

        if brightness_factor < 0:
            brightness_factor = 1.0

        return brightness_factor
    
    def image_transform(self, image):
        brightness_factor = self.get_params(self.brightness_range)
        return adjust_brightness(image, brightness_factor)