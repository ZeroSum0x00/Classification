import cv2
import math
import random
import numbers
import numpy as np

from .resize import INTER_MODE
from .resized_crop import resized_crop
from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def rotate(
    image,
    angle,
    expand=False,
    center=None,
    fill_color=None,
    interpolation="BILINEAR",
):
    rank_size = len(image.shape)
    imgtype = image.dtype
    
    if not is_numpy_image(image):
        raise TypeError("Image should be CV Image. Got {}".format(type(image)))
        
    h, w, _ = image.shape
    point = center or (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=1)

    if fill_color:
        if isinstance(fill_color, int):
            if fill_color == -1:
                color = [random.randint(0, 255) for _ in range(rank_size)]
            else:
                color = [fill_color for _ in range(rank_size)]
        else:
            color = fill_color
    else:
        color = [0 for _ in range(rank_size)]

    if expand:
        if center is None:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - point[0]
            M[1, 2] += (nH / 2) - point[1]

            # perform the actual rotation and return the image
            dst = cv2.warpAffine(image, M, (nW, nH), borderValue=color)
        else:
            xx = []
            yy = []
            for point in (np.array([0, 0, 1]), np.array([w-1, 0, 1]), np.array([w-1, h-1, 1]), np.array([0, h-1, 1])):
                target = M@point
                xx.append(target[0])
                yy.append(target[1])
            nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
            nw = int(math.ceil(max(xx)) - math.floor(min(xx)))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nw - w)/2
            M[1, 2] += (nh - h)/2
            dst = cv2.warpAffine(image, M, (nw, nh), flags=INTER_MODE[interpolation], borderValue=fill_color)
    else:
        dst = cv2.warpAffine(image, M, (w, h), flags=INTER_MODE[interpolation], borderValue=fill_color)
    return dst.astype(imgtype)


class Rotation(BaseTransform):

    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
        interpolation ({CV.Image.NEAREST, CV.Image.BILINEAR, CV.Image.BICUBIC}, optional):
            An optional resampling filter.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(
        self,
        degrees,
        expand=False,
        center=None,
        fill_color=None,
        interpolation="BILINEAR",
    ):
        self.degrees = degrees
        self.expand = expand
        self.center = center
        self.fill_color = fill_color
        self.interpolation = interpolation

    def image_transform(self, image):
        return rotate(
            image=image,
            angle=self.degrees,
            expand=self.expand,
            center=self.center,
            fill_color=self.fill_color,
            interpolation=self.interpolation,
        )


class RandomRotation(BaseRandomTransform):

    """Rotate the image in range angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
        interpolation ({CV.Image.NEAREST, CV.Image.BILINEAR, CV.Image.BICUBIC}, optional):
            An optional resampling filter.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(
        self,
        degrees,
        expand=False,
        center=None,
        fill_color=None,
        interpolation="BILINEAR",
        prob=0.5,
    ):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.expand = expand
        self.center = center
        self.fill_color = fill_color
        self.interpolation = interpolation
        self.prob = prob

    @staticmethod
    def get_params(degrees):
        return random.uniform(degrees[0], degrees[1])

    def image_transform(self, image):
        angle = self.get_params(self.degrees)
        return rotate(
            image=image,
            angle=angle,
            expand=self.expand,
            center=self.center,
            fill_color=self.fill_color,
            interpolation=self.interpolation,
        )
    