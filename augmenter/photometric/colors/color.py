import cv2
import random
import numbers
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image
from ..blends import blend



def color(image, color_factor):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))

    degenerate = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    degenerate = cv2.cvtColor(degenerate, cv2.COLOR_GRAY2RGB)
    return blend(degenerate, image, color_factor)


class Color(BaseTransform):
    def __init__(self, color_factor=1):
        self.color_factor = color_factor

    def image_transform(self, image):
        return color(image, self.color_factor)


class RandomColor(BaseRandomTransform):
    def __init__(self, color_range=1, prob=0.5):
        assert (isinstance(color_range, (int, float)) and color_range > 0) or (isinstance(color_range, collections.Iterable) and len(color_range) == 2)
        self.color_range = color_range
        self.prob = prob

    @staticmethod
    def get_params(color):
        color_factor = 0.0

        if isinstance(color, numbers.Number) and color > 0:
            color_factor = random.uniform(-color, color)
        elif isinstance(color, (tuple, list)):
            color_factor = random.uniform(color[0], color[1])

        if not(0. < color_factor < 1):
            color_factor = 0.0
            
        return color_factor

    def image_transform(self, image):
        color_factor = self.get_params(self.color_range)
        return color(image, color_factor)
    