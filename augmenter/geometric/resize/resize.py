import cv2
import random
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform
from utils.auxiliary_processing import is_numpy_image
from utils.constants import INTER_MODE



def resize(image, size, interpolation='BILINEAR'):
    if not is_numpy_image(image):
        raise TypeError('Image input should be CV Image. Got {}'.format(type(image)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w, c = image.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return image
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(image, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(image, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
    else:
        oh, ow = size
        return cv2.resize(image, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation])


class Resize(BaseTransform):

    """Resize the input image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is ``BILINEAR``
    """

    def __init__(self, size, interpolation='BILINEAR'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def image_transform(self, image):
        return resize(image, self.size, self.interpolation)