import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform



def salt_and_pepper_noise(img, threshold=0.01):
    imgtype = img.dtype
    rnd = np.random.rand(img.shape[0], img.shape[1])
    noisy = img.copy()
    noisy[rnd < threshold / 2] = 0.0
    noisy[rnd > 1 - threshold / 2] = 255.0
    return noisy.astype(imgtype)


class SaltPepperNoise(BaseTransform):
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def image_transform(self, image):
        return salt_and_pepper_noise(image, self.threshold)


class RandomSaltPepperNoise(BaseRandomTransform):
    def __init__(self, threshold_range=0.1, prob=0.5):
        assert isinstance(threshold_range, (int, float)) or (isinstance(threshold_range, collections.Iterable) and len(threshold_range) == 2)
        self.threshold_range = threshold_range
        self.prob = prob

    @staticmethod
    def get_params(threshold):
        if isinstance(threshold, numbers.Number):
            threshold_factor = random.uniform(0, threshold)
        else:
            threshold_factor = random.uniform(threshold[0], threshold[1])
        return threshold_factor

    def image_transform(self, image):
        threshold_factor = self.get_params(self.threshold_range)
        return salt_and_pepper_noise(image, threshold_factor)
    