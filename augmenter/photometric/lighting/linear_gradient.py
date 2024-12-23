import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def linear_gradient(image, orientation="horizontal", edge_brightness=(.1, .3)):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))
    
    assert isinstance(edge_brightness, tuple) and len(edge_brightness) == 2, \
        "Argument edge_brightness should be a tuple with size 2."
    assert 0. < edge_brightness[0] < 1. and 0. < edge_brightness[1] < 1., \
        "Values of an edge_brightness argument should be in the [0, 1] range."
    assert orientation in ['horizontal', 'vertical'], "Unknown orientation value."

    color1 = int(edge_brightness[0] * 255)
    color2 = int(edge_brightness[1] * 255)
    reverse = bool(random.getrandbits(1))

    image = np.int16(image.copy())
    dim = image.shape[1] if orientation == "horizontal" else image.shape[0]
    for i in range(dim):
        coeff = i / float(dim)
        if reverse:
            coeff = 1. - coeff
        diff = int((color2 - color1) * coeff)
        if orientation == "horizontal":
            image[:, i, 0:3] = np.where(image[:, i, 0:3] + color1 + diff < 255,
                                        image[:, i, 0:3] + color1 + diff, 255)
        else:
            image[i, :, 0:3] = np.where(image[i, :, 0:3] + color1 + diff < 255,
                                        image[i, :, 0:3] + color1 + diff, 255)

    return image.astype(np.uint8)


class LinearGradient(BaseTransform):
    def __init__(self,
                 orientation="horizontal", 
                 edge_brightness=(.1, .3)):
        self.orientation     = orientation
        self.edge_brightness = edge_brightness

    def image_transform(self, image):
        return linear_gradient(image, self.orientation, self.edge_brightness)


class RandomLinearGradient(BaseRandomTransform):
    def __init__(self,
                 orientation="horizontal", 
                 edge_brightness=(.1, .3),
                 prob=0.5):
        self.orientation     = orientation
        self.edge_brightness = edge_brightness
        self.prob            = prob

    def image_transform(self, image):
        return linear_gradient(image, self.orientation, self.edge_brightness)