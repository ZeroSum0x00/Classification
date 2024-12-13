import cv2
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def vflip(image):
    if not is_numpy_image(image):
        raise TypeError('Image input should be CV Image. Got {}'.format(type(image)))

    return cv2.flip(image, 0)

    
def hflip(image):
    if not is_numpy_image(image):
        raise TypeError('Image input should be CV Image. Got {}'.format(type(image)))

    return cv2.flip(image, 1)


class Flip(BaseTransform):

    """Flip transformation the given image.

    Args:
        mode ({horizontal, vertical, synthetic}): A flip mode.
    """

    def __init__(self, mode='horizontal'):
        self.mode            = mode
        self.horizontal_list = ['horizontal', 'h']
        self.vertical_list   = ['vertical', 'v']
        self.mix_list        = ['synthetic', 's']

    def image_transform(self, image):
        if self.mode.lower() in self.horizontal_list or self.mode.lower() in self.mix_list:
            rand_point = np.random.randint(0, 2) if self.mode.lower() in self.mix_list else 1
            if rand_point:
                if isinstance(image, (tuple, list)):
                    image = [hflip(img) for img in image]
                else:
                    image = hflip(image)

        if self.mode.lower() in self.vertical_list or self.mode.lower() in self.mix_list:
            rand_point = np.random.randint(0, 2) if self.mode.lower() in self.mix_list else 1
            if rand_point:
                if isinstance(image, (tuple, list)):
                    image = [vflip(img) for img in image]
                else:
                    image = vflip(image)

        return image


class RandomFlip(BaseRandomTransform):

    """Random flip transformation the given image randomly with a given probability.

    Args:
        mode ({horizontal, vertical, synthetic}): A flip mode.
        prob (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, mode='horizontal', prob=0.5):
        self.mode = mode
        self.prob = prob
        self.aug  = Flip(mode=self.mode)

    def image_transform(self, image):
        image = self.aug(image)
        return image


class RandomHorizontalFlip(BaseRandomTransform):

    """Horizontally flip the given image randomly with a given probability.

    Args:
        prob (float): probability of the image being flipped. Default value is 0.5
    """

    def image_transform(self, image):
        return hflip(image)


class RandomVerticalFlip(BaseRandomTransform):

    """Vertically flip the given image randomly with a given probability.

    Args:
        prob (float): probability of the image being flipped. Default value is 0.5
    """

    def image_transform(self, image):
        return vflip(image)