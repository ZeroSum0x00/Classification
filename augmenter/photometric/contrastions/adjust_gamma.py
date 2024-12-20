import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def adjust_gamma(img, gamma, gain=1):
    if not is_numpy_image(img):
        raise TypeError('img should be image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    im = img.astype(np.float32)
    im = 255. * gain * np.power(im / 255., gamma)
    im = im.clip(min=0., max=255.)
    return im.astype(img.dtype)


class AdjustGamma(BaseTransform):
    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain  = gain

    def image_transform(self, image):
        return adjust_gamma(image, self.gamma, self.gain)


class RandomAdjustGamma(BaseRandomTransform):
    def __init__(self, gamma, gain=1, prob=0.5):

        if isinstance(gamma, numbers.Number) and gamma < 0:
            raise ValueError('Gamma should be a non-negative real number')

        self.gamma = gamma
        self.gain  = gain
        self.prob  = prob

    @staticmethod
    def get_params(gamma):
        gamma_factor = 1.0

        if isinstance(gamma, numbers.Number) and gamma > 0:
            gamma_factor = random.uniform(0, gamma)
        else:
            if gamma[0] > 0 and gamma[1] > 0:  
                gamma_factor = random.uniform(gamma[0], gamma[1])

        if gamma_factor < 0:
            gamma_factor = 1.0

        return gamma_factor
    
    def image_transform(self, image):
        gamma_factor = self.get_params(self.gamma)
        return adjust_gamma(image, gamma_factor, gain=self.gain)