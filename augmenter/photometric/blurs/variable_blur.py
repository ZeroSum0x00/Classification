import cv2
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from ..lighting import linear_gradient, radial_gradient
from utils.auxiliary_processing import is_numpy_image



def variable_blur(image, modes=("linear", "radial"), ksize_norm=.2):

    def linear_mask(image):
      mask = np.zeros(image.shape, dtype=np.uint8)
      edge1 = random.uniform(0.7, 1.)
      edge2 = random.uniform(0, .3)
      return linear_gradient(
          mask,
          orientation=random.choice(["horizontal", "vertical"]),
          edge_brightness=(edge1, edge2),
      )

    def radial_mask(image):
        max_dim = max(image.shape)
        mask = np.zeros(image.shape, dtype=np.uint8)
        return radial_gradient(
            mask,
            inner_color=255,
            outer_color=10,
            max_distance=max_dim * random.uniform(.35, .65),
        )

    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))

    for elem in modes:
        assert elem in ["linear", "radial"]
    
    k_size = int(min(image.shape[:2]) * ksize_norm)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size
    if k_size <= 2:
        return image

    image_blurred = cv2.blur(image.copy(), ksize=(k_size, k_size))

    mode = random.choice(modes)

    if mode == "linear":
        mask = linear_mask(image)
    elif mode == "radial":
        mask = radial_mask(image)

    image = image.astype(np.float32)
    image_blurred = image_blurred.astype(np.float32)
    mask = mask.astype(np.float32) / 255

    image = cv2.multiply(mask, image).astype(np.uint8)
    image_blurred = cv2.multiply(1.0 - mask, image_blurred).astype(np.uint8)
    return cv2.add(image, image_blurred)


class VariableBlur(BaseTransform):
    def __init__(self, modes=("linear", "radial"), ksize_norm=.2):
        self.modes = modes
        self.ksize_norm = ksize_norm

    def image_transform(self, image):
        return variable_blur(image, self.modes, self.ksize_norm)


class RandomVariableBlur(BaseRandomTransform):
    def __init__(self, modes=("linear", "radial"), ksize_norm=.2, prob=0.5):
        self.modes = modes
        self.ksize_norm = ksize_norm
        self.prob = prob

    def image_transform(self, image):
        return variable_blur(image, self.modes, self.ksize_norm)
    