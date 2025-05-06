import cv2
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def desaturate(image, percent):
    if not(0. <= percent <= 1.):
        raise ValueError("percent is not in [0, 1].".format(percent))

    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))

    if percent == 0:
        return img

    img = image.copy()

    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # desaturate
    s_desat = cv2.multiply(s, 1 - percent).astype(np.uint8)
    hsv_new = cv2.merge([h, s_desat, v])
    bgr_desat = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    # create 1D LUT for green
    # (120 out of 360) = (60 out of 180)  +- 25
    lut = np.zeros((1, 256), dtype=np.uint8)
    white = np.full((1, 50), 255, dtype=np.uint8)
    lut[0:1, 35:85] = white

    # apply lut to hue channel as mask
    mask = cv2.LUT(h, lut)
    mask = mask.astype(np.float32) / 255
    mask = cv2.merge([mask, mask, mask])

    # mask bgr_desat and img
    img = mask * bgr_desat + (1 - mask) * img
    img = img.clip(0,255).astype(np.uint8)
    return img


class Desaturate(BaseTransform):
    def __init__(self, percent):
        self.percent = percent

    def image_transform(self, image):
        return desaturate(image, self.percent)


class RandomDesaturate(BaseRandomTransform):
    def __init__(self, percent, prob=0.5):
        self.percent = percent
        self.prob = prob

    @staticmethod
    def get_params(percent):
        percent_factor = 1.

        if isinstance(percent, numbers.Number) and percent > 0:
            percent_factor = random.uniform(0, percent)
        elif isinstance(percent, (tuple, list)):
            percent_factor = random.uniform(percent[0], percent[1])

        if not(0 <= percent_factor <= 0.5):
            percent_factor = 1.
            
        return percent_factor

    def image_transform(self, image):
        percent_factor = self.get_params(self.percent)
        return desaturate(image, percent_factor)
    