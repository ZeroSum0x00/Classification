import cv2
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def shear_x(image, factor):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    M = np.array([[1, factor, 0], [0, 1, 0]], dtype=np.float32) 
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def shear_y(image, factor):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    M = np.array([[1, 0, 0], [factor, 1, 0]], dtype=np.float32) 
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def shear(image, factor_x, factor_y):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))
    
    M = np.array([[1, factor_x, 0], [factor_y, 1, 0]], dtype=np.float32) 
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


class ShearX(BaseTransform):
    def __init__(self, factor):
        self.factor = factor

    def image_transform(self, image):
        return shear_x(image, self.factor)


class RandomShearX(BaseRandomTransform):
    def __init__(self, factor, prob=0.5):
        self.factor = factor
        self.prob   = prob

    @staticmethod
    def get_params(factor):
        shear_factor = 0.0

        if isinstance(factor, numbers.Number) and factor > 0:
            shear_factor = random.uniform(-factor, factor)
        elif isinstance(factor, (tuple, list)):
            shear_factor = random.uniform(factor[0], factor[1])

        return shear_factor

    def image_transform(self, image):
        shear_factor = self.get_params(self.factor)
        return shear_x(image, shear_factor)


class ShearY(BaseTransform):
    def __init__(self, factor):
        self.factor = factor

    def image_transform(self, image):
        return shear_y(image, self.factor)


class RandomShearY(BaseRandomTransform):
    def __init__(self, factor, prob=0.5):
        self.factor = factor
        self.prob   = prob

    @staticmethod
    def get_params(factor):
        shear_factor = 0.0

        if isinstance(factor, numbers.Number) and factor > 0:
            shear_factor = random.uniform(-factor, factor)
        elif isinstance(factor, (tuple, list)):
            shear_factor = random.uniform(factor[0], factor[1])

        return shear_factor

    def image_transform(self, image):
        shear_factor = self.get_params(self.factor)
        return shear_y(image, shear_factor)


class Shear(BaseTransform):
    def __init__(self, factor_x, factor_y):
        self.factor_x = factor_x
        self.factor_y = factor_y

    def image_transform(self, image):
        return shear(image, self.factor_x, self.factor_y)


class RandomShear(BaseRandomTransform):
    def __init__(self, factor_x, factor_y, prob=0.5):
        self.factor_x = factor_x
        self.factor_y = factor_y
        self.prob     = prob

    @staticmethod
    def get_params(factor):
        shear_factor = 0.0

        if isinstance(factor, numbers.Number) and factor > 0:
            shear_factor = random.uniform(-factor, factor)
        elif isinstance(factor, (tuple, list)):
            shear_factor = random.uniform(factor[0], factor[1])

        return shear_factor

    def image_transform(self, image):
        shear_factor = self.get_params(self.factor)
        return shear_x(image, shear_factor)