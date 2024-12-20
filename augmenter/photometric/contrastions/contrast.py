import cv2
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image
from ..blends import blend


def contrast(image, contrast_factor):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    if contrast_factor < 0:
        raise ValueError('Gamma should be a non-negative real number')

    degenerate = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist, _ = np.histogram(degenerate, bins=256, range=[0, 255])
    mean = np.sum(hist) / 256.0
    degenerate = np.ones_like(degenerate, dtype=np.float32) * mean
    degenerate = np.clip(degenerate, 0.0, 255.0)
    degenerate = degenerate.astype(np.uint8)
    degenerate = cv2.cvtColor(degenerate, cv2.COLOR_GRAY2BGR)
    return blend(degenerate, image, contrast_factor)


class Contrast(BaseTransform):
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def image_transform(self, image):
        return contrast(image, self.contrast_factor)


class RandomContrast(BaseRandomTransform):
    def __init__(self, contrast_range, prob=0.5):

        if isinstance(contrast_range, numbers.Number) and contrast_range < 0:
            raise ValueError('Contrast factor should be a non-negative real number')

        self.contrast_range = contrast_range
        self.prob           = prob

    @staticmethod
    def get_params(factor):
        contrast_factor = 1.0

        if isinstance(factor, numbers.Number) and factor > 0:
            contrast_factor = random.uniform(0, factor)
        else:
            if factor[0] > 0 and factor[1] > 0:  
                contrast_factor = random.uniform(factor[0], factor[1])

        if contrast_factor < 0:
            contrast_factor = 1.0

        return contrast_factor
    
    def image_transform(self, image):
        contrast_factor = self.get_params(self.contrast_range)
        return contrast(image, contrast_factor)