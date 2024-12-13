import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def solarize_add(image, add_value, threshold=128):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(img)))

    img = image.copy()
    idx = img < threshold
    img[idx] = np.minimum(image[idx] + add_value, 255)
    return img

class SolarizeAdd(BaseTransform):
    def __init__(self, add_value, threshold=128):
        self.add_value = add_value
        self.threshold = threshold

    def image_transform(self, image):
        return solarize_add(image, self.add_value, self.threshold)


class RandomSolarizeAdd(BaseRandomTransform):
    def __init__(self, add_value, threshold=128, prob=0.5):
        self.add_value = add_value
        self.threshold = threshold
        self.prob      = prob

    def image_transform(self, image):
        return solarize_add(image, self.add_value, self.threshold)