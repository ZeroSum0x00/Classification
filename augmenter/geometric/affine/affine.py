import cv2
import math
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from ..resize import INTER_MODE
from utils.auxiliary_processing import is_numpy_image


def affine(image, angle=0, translate=(0, 0), scale=1, shear=0, resample='BILINEAR', fillcolor=(0, 0, 0)):
    imgtype = image.dtype
    if not is_numpy_image(image):
        raise TypeError('img should be CV Image. Got {}'.format(type(image)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"
    gray_scale = False

    if len(image.shape) == 2:
        gray_scale = True
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    rows, cols, _ = image.shape
    center = (cols * 0.5, rows * 0.5)
    angle = math.radians(angle)
    shear = math.radians(shear)
    M00 = math.cos(angle) * scale
    M01 = -math.sin(angle + shear) * scale
    M10 = math.sin(angle) * scale
    M11 = math.cos(angle + shear) * scale
    M02 = center[0] - center[0] * M00 - center[1] * M01 + translate[0]
    M12 = center[1] - center[0] * M10 - center[1] * M11 + translate[1]
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)
    dst_img = cv2.warpAffine(image, 
                             affine_matrix, 
                             (cols, rows), 
                             flags=INTER_MODE[resample],
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=fillcolor)
    if gray_scale:
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2GRAY)
    return dst_img.astype(imgtype)


class Affine(BaseTransform):

    """Affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(self, degrees, translate=(0, 0), scale=1, shear=0, resample='BILINEAR', fillcolor=(0, 0, 0)):
        self.degrees   = degrees
        self.translate = translate
        self.scale     = scale
        self.shear     = shear
        self.resample  = resample
        self.fillcolor = fillcolor

    def image_transform(self, image):
        return affine(image, self.degrees, self.translate, self.scale, self.shear, self.resample, self.fillcolor)


class RandomAffine(BaseRandomTransform):

    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(self, degrees=0, translate=None, scale=None, shear=None, resample='BILINEAR', fillcolor=0, prob=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample  = resample
        self.fillcolor = fillcolor
        self.prob      = prob

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[1]
            max_dy = translate[1] * img_size[0]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def image_transform(self, image):
        angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear, image.shape)
        return affine(image, angle, translations, scale, shear, resample=self.resample, fillcolor=self.fillcolor)