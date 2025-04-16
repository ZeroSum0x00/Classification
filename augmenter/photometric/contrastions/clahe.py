import cv2
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    assert (image.dtype == np.uint8)
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


class CLAHE(BaseTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def image_transform(self, image):
        return clahe(image, self.clip_limit, self.tile_grid_size)


class RandomCLAHE(BaseRandomTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), prob=0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.prob = prob

    def image_transform(self, image):
        return clahe(image, self.clip_limit, self.tile_grid_size)
