import cv2
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image



def camera_flare(image, radius=0.5, alpha=0.8):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))
        
    assert 0 < radius <= 1

    im_height, im_width = image.shape[:2]
    pos_x, pos_y = random.randint(0, im_width), random.randint(0, im_height)
    avg_dim = (im_height + im_width) / 2
    radius = int(avg_dim * radius)

    # white circle
    circle = np.zeros((im_height, im_width, 3), np.uint8)

    cv2.circle(circle, (pos_x, pos_y), radius, (255, 255, 255), -1)
    circle = cv2.blur(
        circle,
        (int(random.uniform(.15, .25) * avg_dim), int(random.uniform(.15, .25) * avg_dim)))

    dst = cv2.addWeighted(image, 1.0, circle, alpha, 0.0)
    return cv2.resize(dst, (im_width, im_height), interpolation=cv2.INTER_CUBIC)


class CameraFlare(BaseTransform):
    def __init__(self, radius=0.5, alpha=0.8):
        self.radius = radius
        self.alpha = alpha

    def image_transform(self, image):
        return camera_flare(image, self.radius, self.alpha)

class RandomCameraFlare(BaseRandomTransform):
    def __init__(self, radius=0.5, alpha=0.8, prob=0.5):
        self.radius = radius
        self.alpha = alpha
        self.prob = prob

    def image_transform(self, image):
        return camera_flare(image, self.radius, self.alpha)
    