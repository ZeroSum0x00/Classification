import cv2
import math
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image



def radial_gradient(
    image,
    inner_color=150,
    outer_color=30,
    center=None,
    max_distance=None,
    rect=False,
    random_distance=False,
):
  
    def apply_radial(
        image,
        center,
        max_distance,
        inner_color,
        outer_color,
        rect=False,
    ):
        tmp = np.full(image.shape, outer_color, dtype=np.uint8)
        tmp_height, tmp_width = tmp.shape[:2]
        kernel = None

        left = max(0, 0 - (center[0] - max_distance))
        top = max(0, 0 - (center[1] - max_distance))
        right = max(0, (center[0] + max_distance) - tmp_width)
        bottom = max(0, (center[1] + max_distance) - tmp_height)
        tmp = cv2.copyMakeBorder(tmp, top, bottom, left, right, cv2.BORDER_CONSTANT)

        if rect:
            if random.getrandbits(1):
                dist = random.randint(10, int(.2 * tmp_width))
                cv2.rectangle(
                    tmp,
                    (center[0] - dist, 0),
                    (center[0] + dist, tmp_height),
                    inner_color,
                    thickness=cv2.FILLED,
                )
                k_size = dist if dist % 2 == 1 else dist - 1
                kernel = (k_size, 1)
            else:
                dist = random.randint(10, int(.2 * tmp_height))
                cv2.rectangle(
                    tmp,
                    (0, center[1] - dist),
                    (tmp_width, center[1] + dist),
                    inner_color,
                    thickness=cv2.FILLED,
                )
                k_size = dist if dist % 2 == 1 else dist - 1
                kernel = (1, k_size)
        else:
            cv2.circle(
                tmp,
                (center[0] + left, center[1] + top),
                int(max_distance / 1.5),
                inner_color,
                thickness=cv2.FILLED,
            )

        kernel = kernel if kernel else (max_distance, max_distance)
        tmp = cv2.blur(tmp, kernel, borderType=cv2.BORDER_CONSTANT)
        tmp = tmp[top:tmp.shape[0] - bottom, left:tmp.shape[1] - right]

        return np.clip(image.astype(np.uint16) + tmp.astype(np.uint16), 0, 255).astype(np.uint8)

    if not is_numpy_image(image):
        raise TypeError("image should be image. Got {}".format(type(image)))
    
    im_height, im_width, im_depth = image.shape
    inner_color = im_depth * [inner_color]
    outer_color = im_depth * [outer_color]

    if center is None:
        center = random.randint(0, im_height), random.randint(0, im_width)

    if not rect:
        if max_distance is None:
            if random_distance:
                size = max(im_width, im_height)
                max_distance = size * random.uniform(.1, .3)
            else:
                max_distance = 0
                corners = [(0, 0), (im_height, 0), (0, im_width), (im_height, im_width)]
                for corner in corners:
                    distance = math.sqrt((corner[0] - center[0])**2 +
                                         (corner[1] - center[1])**2)
                    max_distance = max(distance, max_distance)

    return apply_radial(
        image=image,
        center=center,
        max_distance=int(max_distance),
        inner_color=inner_color,
        outer_color=outer_color,
        rect=rect,
    )
    

class RadialGradient(BaseTransform):
    def __init__(
        self,
        inner_color=150,
        outer_color=30,
        center=None,
        max_distance=None,
        rect=False,
        random_distance=False,
    ):
        self.inner_color = inner_color
        self.outer_color = outer_color
        self.center = center
        self.max_distance = max_distance
        self.rect = rect
        self.random_distance = random_distance

    def image_transform(self, image):
        return radial_gradient(
            image=image,
            inner_color=self.inner_color,
            outer_color=self.outer_color,
            center=self.center,
            max_distance=self.max_distance,
            rect=self.rect,
            random_distance=self.random_distance,
        )


class RandomRadialGradient(BaseRandomTransform):
    def __init__(
        self,
        inner_color=150,
        outer_color=30,
        center=None,
        max_distance=None,
        rect=False,
        random_distance=False,
        prob=0.5
    ):
        self.inner_color = inner_color
        self.outer_color = outer_color
        self.center = center
        self.max_distance = max_distance
        self.rect = rect
        self.random_distance = random_distance
        self.prob = prob

    def image_transform(self, image):
        return radial_gradient(
            image=image,
            inner_color=self.inner_color,
            outer_color=self.outer_color,
            center=self.center,
            max_distance=self.max_distance,
            rect=self.rect,
            random_distance=self.random_distance,
        )
    