import cv2
import numpy as np

from utils.auxiliary_processing import is_numpy_image


# def blend(image1, image2, ratio=0.8):

#     if not is_numpy_image(image1):
#         raise TypeError('img1 should be image. Got {}'.format(type(image1)))

#     if not is_numpy_image(image2):
#         raise TypeError('img2 should be image. Got {}'.format(type(image2)))
    
#     if ratio == 0.0:
#         return image1
#     if ratio == 1.0:
#         return image2

#     if ratio > 0.0 and ratio < 1.0:
#         return cv2.addWeighted(image1, 1 - ratio, image2, ratio, 0)
#     else:
#         return image1


def blend(image1, image2, factor):
    if not is_numpy_image(image1):
        raise TypeError('img1 should be image. Got {}'.format(type(image1)))

    if not is_numpy_image(image2):
        raise TypeError('img2 should be image. Got {}'.format(type(image2)))
    
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1.astype(np.float32) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp.astype(np.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return np.clip(temp, 0.0, 255.0).astype(np.uint8)


class Blend:
    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def __call__(self, image1, image2):
        return blend(image1, image2, factor=self.ratio)