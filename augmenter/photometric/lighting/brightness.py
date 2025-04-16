import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image
from ..blends import blend


def brightness(image, factor):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    if factor < 0:
        raise ValueError('Brightness should be a non-negative real number')

    degenerate = np.zeros_like(image)
    return blend(degenerate, image, factor)


class Brightness(BaseTransform):
    def __init__(self, factor):
        self.factor = factor

    def image_transform(self, image):
        return brightness(image, self.factor)


class RandomBrightness(BaseRandomTransform):
    def __init__(self, factor, prob=0.5):

        if isinstance(factor, numbers.Number) and factor < 0:
            raise ValueError('Brightness factor should be a non-negative real number')

        self.factor = factor
        self.prob = prob

    @staticmethod
    def get_params(factor):
        brightness_factor = 1.0

        if isinstance(factor, numbers.Number) and factor > 0:
            brightness_factor = random.uniform(0, factor)
        else:
            if factor[0] > 0 and factor[1] > 0:  
                brightness_factor = random.uniform(factor[0], factor[1])

        if brightness_factor < 0:
            brightness_factor = 1.0

        return brightness_factor
    
    def image_transform(self, image):
        brightness_factor = self.get_params(self.factor)
        return brightness(image, brightness_factor)