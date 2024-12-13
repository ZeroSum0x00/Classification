import cv2

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def to_grayscale(image, out_channels=1):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    if out_channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif out_channels == 3:
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return image

class Grayscale(BaseTransform):
    def __init__(self, out_channels=1):
        self.out_channels = out_channels

    def image_transform(self, image):
        return to_grayscale(image, self.out_channels)


class RandomGrayscale(BaseRandomTransform):
    def __init__(self, out_channels=3, prob=0.5):
        self.out_channels = out_channels
        self.prob         = prob

    def image_transform(self, image):
        return to_grayscale(image, self.out_channels)