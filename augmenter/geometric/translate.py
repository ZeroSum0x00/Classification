import cv2
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def translate_x(image, pct, mode="rel"):
    pixels = pct * image.shape[1] if mode == "rel" else pct
    M = np.array([[1, 0, pixels], [0, 1, 0]], dtype=np.float32) 
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def translate_y(image, pct, mode="rel"):
    pixels = pct * image.shape[0] if mode == "rel" else pct
    M = np.array([[1, 0, 0], [0, 1, pixels]], dtype=np.float32) 
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


class TranslateX(BaseTransform):
    def __init__(self, pct, mode="rel"):
        self.pct = pct
        self.mode = mode

    def image_transform(self, image):
        return translate_x(image, self.pct, self.mode)


class RandomTranslateX(BaseRandomTransform):
    def __init__(self, pct, mode="rel", prob=0.5):
        self.pct = pct
        self.mode = mode
        self.prob = prob

    @staticmethod
    def get_params(factor):
        translate_factor = 0.0

        if isinstance(factor, numbers.Number) and factor > 0:
            translate_factor = random.uniform(-factor, factor)
        elif isinstance(factor, (tuple, list)):
            translate_factor = random.uniform(factor[0], factor[1])

        return translate_factor

    def image_transform(self, image):
        translate_factor = self.get_params(self.pct)
        return translate_x(image, translate_factor, mode=self.mode)


class TranslateY(BaseTransform):
    def __init__(self, pct, mode="rel"):
        self.pct = pct
        self.mode = mode

    def image_transform(self, image):
        return translate_y(image, self.pct, self.mode)


class RandomTranslateY(BaseRandomTransform):
    def __init__(self, pct, mode="rel", prob=0.5):
        self.pct = pct
        self.mode = mode
        self.prob = prob

    @staticmethod
    def get_params(factor):
        translate_factor = 0.0

        if isinstance(factor, numbers.Number) and factor > 0:
            translate_factor = random.uniform(-factor, factor)
        elif isinstance(factor, (tuple, list)):
            translate_factor = random.uniform(factor[0], factor[1])

        return translate_factor

    def image_transform(self, image):
        translate_factor = self.get_params(self.pct)
        return translate_y(image, translate_factor, mode=self.mode)
    