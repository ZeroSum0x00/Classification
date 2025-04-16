import cv2
import math
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def erasing(
    image, 
    min_area=0.02,
    max_area=1/3, 
    min_aspect=0.3,
    max_aspect=None, 
    min_count=1,
    max_count=1, 
    fill_color=0,
    mode='constant',
):

    def _get_pixels(img, size, mode, fill_color):
        if mode == "random":
            fill_color = [random.randint(0, 255) for _ in range(img.shape[-1])]
        return np.full(size, fill_value=fill_color)

    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))
        
    max_aspect = max_aspect or 1 / min_aspect
    img = image.copy()
    area = img.shape[0] * img.shape[1]
    count = min_count if min_count == max_count else random.randint(min_count, max_count)
    log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
    for _ in range(count):
        for attempt in range(10):
            target_area = random.uniform(min_area, max_area) * area / count
            aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if h < img.shape[0] and w < img.shape[1]:
                top = random.randint(0, img.shape[0] - h)
                left = random.randint(0, img.shape[1] - w)
                img[top:top + h, left:left + w, :] = _get_pixels(img, (h, w, img.shape[-1]), mode, fill_color)
                break
    return img


class Erasing(BaseTransform):

    def __init__(
        self, 
        min_area=0.02,
        max_area=1/3, 
        min_aspect=0.3,
        max_aspect=None, 
        min_count=1,
        max_count=1, 
        fill_color=0,
        mode='constant',
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_count = min_count
        self.max_count = max_count
        self.fill_color = fill_color
        self.mode = mode

    def image_transform(self, image):
        return erasing(
            image=image, 
            min_area=self.min_area,
            max_area=self.max_area, 
            min_aspect=self.min_aspect,
            max_aspect=self.max_aspect,
            min_count=self.min_count,
            max_count=self.max_count,
            fill_color=self.fill_color,
            mode=self.mode,
        )


class RandomErasing(BaseRandomTransform):

    def __init__(
        self, 
        min_area=0.02,
        max_area=1/3, 
        min_aspect=0.3,
        max_aspect=None, 
        min_count=1,
        max_count=1, 
        fill_color=0,
        mode='constant',
        prob=0.5,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_count = min_count
        self.max_count = max_count
        self.fill_color = fill_color
        self.mode = mode
        self.prob = prob

    def image_transform(self, image):
        return erasing(
            image=image, 
            min_area=self.min_area,
            max_area=self.max_area, 
            min_aspect=self.min_aspect,
            max_aspect=self.max_aspect,
            min_count=self.min_count,
            max_count=self.max_count,
            fill_color=self.fill_color,
            mode=self.mode,
        )