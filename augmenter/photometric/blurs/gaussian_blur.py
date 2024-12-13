import cv2

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def gaussian_blur(image, ksize_norm=.4, sigma=5, direction=None):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))
        
    assert direction in ('horizontal', 'vertical', None)

    k_size = int(min(image.shape[:2]) * ksize_norm)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size
    if k_size <= 2:
        return image

    if direction == "horizontal":
        return cv2.GaussianBlur(image, (k_size, 1), sigmaX=sigma, sigmaY=sigma)
    elif direction == "vertical":
        return cv2.GaussianBlur(image, (1, k_size), sigmaX=sigma, sigmaY=sigma)
    else:
        return cv2.GaussianBlur(image, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)


class GaussianBlur(BaseTransform):
    def __init__(self, ksize_norm=.4, sigma=5, direction=None):
        self.ksize_norm = ksize_norm
        self.sigma      = sigma
        self.direction  = direction

    def image_transform(self, image):
        return gaussian_blur(image, self.ksize_norm, self.sigma, self.direction)


class RandomGaussianBlur(BaseRandomTransform):
    def __init__(self, ksize_norm=.4, sigma=5, direction=None, prob=0.5):
        self.ksize_norm = ksize_norm
        self.sigma      = sigma
        self.direction  = direction
        self.prob       = prob

    def image_transform(self, image):
        return gaussian_blur(image, self.ksize_norm, self.sigma, self.direction)
 