import cv2
import math
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from ..resize import INTER_MODE


def affine6(image, anglez=0, translate=(0, 0), scale=(1, 1), shear=0, interpolation='BILINEAR', fill_color=(0, 0, 0)):
    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
        "Argument translate should be a list or tuple of length 2"

    imgtype = image.dtype
    gray_scale = False

    if len(image.shape) == 2:
        gray_scale = True
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rows, cols, _ = image.shape
    centery = rows * 0.5
    centerx = cols * 0.5

    alpha = math.radians(shear)
    beta = math.radians(anglez)

    lambda1 = scale[0]
    lambda2 = scale[1]

    tx = translate[0]
    ty = translate[1]

    sina = math.sin(alpha)
    cosa = math.cos(alpha)
    sinb = math.sin(beta)
    cosb = math.cos(beta)

    M00 = cosb * (lambda1 * cosa**2 + lambda2 * sina**2) - sinb * (lambda2 - lambda1) * sina * cosa
    M01 = - sinb * (lambda1 * sina**2 + lambda2 * cosa**2) + cosb * (lambda2 - lambda1) * sina * cosa

    M10 = sinb * (lambda1 * cosa**2 + lambda2 * sina**2) + cosb * (lambda2 - lambda1) * sina * cosa
    M11 = + cosb * (lambda1 * sina**2 + lambda2 * cosa**2) + sinb * (lambda2 - lambda1) * sina * cosa
    M02 = centerx - M00 * centerx - M01 * centery + tx
    M12 = centery - M10 * centerx - M11 * centery + ty
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)

    dst_img = cv2.warpAffine(image, 
                             affine_matrix, 
                             (cols, rows), 
                             flags=INTER_MODE[interpolation],
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=fill_color)
    if gray_scale:
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2GRAY)
    return dst_img.astype(imgtype)


class Affine6(BaseTransform):

    """Affine transformation of the image keeping center invariant

    Args:
        anglez (sequence or float or int): Range of rotate to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-anglez, +anglez). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int): Range of shear to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-shear, +shear). Set to 0 to desactivate shear.
        interpolation ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
        fill_color (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(self, anglez, translate=(0, 0), scale=1, shear=0, interpolation='BILINEAR', fill_color=(0, 0, 0)):
        self.anglez        = anglez
        self.translate     = translate
        self.scale         = scale
        self.shear         = shear
        self.interpolation = interpolation
        self.fill_color    = fill_color

    def image_transform(self, image):
        return affine6(image, self.anglez, self.translate, self.scale, self.shear, self.interpolation, self.fill_color)


class RandomAffine6(BaseRandomTransform):

    """Random affine transformation of the image keeping center invariant

    Args:
        anglez (sequence or float or int): Range of rotate to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-anglez, +anglez). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int): Range of shear to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-shear, +shear). Set to 0 to desactivate shear.
        interpolation ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
        fill_color (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    
    def __init__(self, anglez=0, translate=(0, 0), scale=(1, 1), shear=0, interpolation='BILINEAR', fill_color=(0, 0, 0), prob=0.5):
        if isinstance(anglez, numbers.Number):
            if anglez < 0:
                raise ValueError("If anglez is a single number, it must be positive.")
            self.anglez = (-anglez, anglez)
        else:
            assert isinstance(anglez, (tuple, list)) and len(anglez) == 2, \
                "anglez should be a list or tuple and it must be of length 2."
            self.anglez = anglez

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

        if isinstance(shear, numbers.Number):
            if shear < 0:
                raise ValueError("If shear is a single number, it must be positive.")
            self.shear = (-shear, shear)
        else:
            assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                "shear should be a list or tuple and it must be of length 2."
            self.shear = shear
            
        self.interpolation = interpolation
        self.fill_color    = fill_color
        self.prob          = prob

    @staticmethod
    def get_params(img_size, anglez_range=(0, 0), translate=(0, 0), scale_ranges=(1, 1), shear_range=(0, 0)):
        angle = random.uniform(anglez_range[0], anglez_range[1])
        shear = random.uniform(shear_range[0], shear_range[1])

        max_dx = translate[0] * img_size[1]
        max_dy = translate[1] * img_size[0]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))

        scale = (random.uniform(1 / scale_ranges[0], scale_ranges[0]),
                 random.uniform(1 / scale_ranges[1], scale_ranges[1]))
        return angle, translations, scale, shear

    def image_transform(self, image):
        ret = self.get_params(self.anglez, self.translate, self.scale, self.shear, image.shape)
        return affine6(image, *ret, interpolation=self.interpolation, fill_color=self.fill_color)