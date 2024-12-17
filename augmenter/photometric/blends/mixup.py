import cv2
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def mixup(images, target_size=(224, 224, 3), main_image_ratio=0.8):
    new_image   = 0.0
    n_sample    = len(images)
    remaining_object_ratio = (1. - main_image_ratio) / (n_sample - 1)
    
    for idx, image in enumerate(images):
        if not is_numpy_image(images):
            raise TypeError('img should be image. Got {}'.format(type(images)))

        h, w, _ = image.shape

        if h != target_size[0] or w != target_size[1]:
            image = cv2.resize(image, target_size[:-1])

        if idx == 0:
            current_ratio = main_image_ratio
                    
        else:
            current_ratio = remaining_object_ratio

        new_image += np.array(image, np.float32) * current_ratio
    return new_image.astype(np.uint8)


class Mixup(BaseTransform):
    def __init__(self, target_size=(416, 416, 3), main_image_ratio=0.5):
        self.target_size      = target_size
        self.main_image_ratio = main_image_ratio

    def image_transform(self, images):
        return mixup(images, target_size=self.target_size, main_image_ratio=self.main_image_ratio)


class RandomMixup(BaseRandomTransform):
    def __init__(self, target_size=(416, 416, 3), main_image_ratio=0.5, prob=0.5):
        self.target_size      = target_size
        self.main_image_ratio = main_image_ratio
        self.prob             = prob

    def image_transform(self, images):
        return mixup(images, target_size=self.target_size, main_image_ratio=self.main_image_ratio)