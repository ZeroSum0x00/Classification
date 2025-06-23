import cv2
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image



def flashlight(image, radius=0.5, alpha=0.8, bg_darkness=100):
    def decrease_brightness(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
        
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))
        
    im_height, im_width = image.shape[:2]
    pos_x = random.randint(int(1 / 4 * im_width), int(3 / 4 * im_width))
    pos_y = random.randint(int(1 / 4 * im_height), int(3 / 4 * im_height))
    min_wh = min(im_width, im_height)
    max_wh = max(im_width, im_height)
    radius = int(random.randint(min_wh, max_wh) * radius)

    k = random.uniform(1.5, 5.)
    blur_kernel_size = (int(radius / k), int(radius / k))
    darkened_image = decrease_brightness(image, value=bg_darkness)

    # create white circle on black background
    torchlight = np.zeros((im_height, im_width, 3), np.uint8)
    cv2.circle(torchlight, (pos_x, pos_y), radius, (255, 255, 255), -1)

    blurred_torchlight = cv2.blur(torchlight, blur_kernel_size)
    final_image = cv2.addWeighted(darkened_image, 1.0, blurred_torchlight, alpha, 0.0)
    return final_image


class Flashlight(BaseTransform):
    def __init__(self, radius=0.5, alpha=0.8, bg_darkness=100):
        self.radius = radius
        self.alpha = alpha
        self.bg_darkness = bg_darkness

    def image_transform(self, image):
        return flashlight(image, self.radius, self.alpha, self.bg_darkness)


class RandomFlashlight(BaseRandomTransform):
    def __init__(self, radius=0.5, alpha=0.8, bg_darkness=100, prob=0.5):
        self.radius = radius
        self.alpha = alpha
        self.bg_darkness = bg_darkness
        self.prob = prob

    def image_transform(self, image):
        return flashlight(image, self.radius, self.alpha, self.bg_darkness)
    