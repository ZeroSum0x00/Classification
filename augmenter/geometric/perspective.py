import cv2
import math
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from .resize import INTER_MODE


def perspective(
    image,
    fov=45,
    anglex=0,
    angley=0,
    anglez=0,
    translate=(0, 0),
    scale=(1, 1),
    shear=0,
    interpolation="BILINEAR",
    fill_color=(0, 0, 0),
):
    imgtype = image.dtype
    gray_scale = False

    if len(image.shape) == 2:
        gray_scale = True
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    h, w, _ = image.shape
    centery = h * 0.5
    centerx = w * 0.5

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
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12], [0, 0, 1]], dtype=np.float32)
    # -------------------------------------------------------------------------------
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(math.radians(fov / 2))

    radx = math.radians(anglex)
    rady = math.radians(angley)

    sinx = math.sin(radx)
    cosx = math.cos(radx)
    siny = math.sin(rady)
    cosy = math.cos(rady)

    r = np.array([[cosy, 0, -siny, 0],
                  [-siny * sinx, cosx, -sinx * cosy, 0],
                  [cosx * siny, sinx, cosx * cosy, 0],
                  [0, 0, 0, 1]])

    pcenter = np.array([centerx, centery, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    perspective_matrix = cv2.getPerspectiveTransform(org, dst)
    total_matrix = perspective_matrix @ affine_matrix

    result_img = cv2.warpPerspective(image,
                                     total_matrix,
                                     (w, h),
                                     flags=INTER_MODE[interpolation],
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=fill_color)
    if gray_scale:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
    return result_img.astype(imgtype)


class Perspective(BaseTransform):

    """Perspective transformation of the image keeping center invariant
        Args:
            fov(float): range of wide angle = 90+-fov
            anglex (sequence or float or int): Range of degrees rote around X axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            angley (sequence or float or int): Range of degrees rote around Y axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            anglez (sequence or float or int): Range of degrees rote around Z axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            translate (tuple, optional): tuple of maximum absolute fraction for horizontal
                and vertical translations. For example translate=(a, b), then horizontal shift
                is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
                randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
            scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
                randomly sampled from the range a <= scale <= b. Will keep original scale by default.
            shear (sequence or float or int): Range of degrees for shear rote around axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            interpolation ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
            fill_color (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
        """

    def __init__(
        self,
        fov=45,
        anglex=0,
        angley=0,
        anglez=0,
        translate=(0, 0),
        scale=(1, 1),
        shear=0,
        interpolation="BILINEAR",
        fill_color=(0, 0, 0),
    ):
        self.fov    = fov
        self.anglex = anglex
        self.angley = angley
        self.anglez = anglez

        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "translate should be a list or tuple and it must be of length 2."
        assert all([0.0 <= i <= 1.0 for i in translate]), "translation values should be between 0 and 1"
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            assert all([s > 0 for s in scale]), "scale values should be positive"
        self.scale = scale

        self.shear     = shear
        self.interpolation  = interpolation
        self.fill_color = fill_color

    def image_transform(self, image):
        return perspective(
            image=image,
            fov=self.fov,
            anglex=self.anglex,
            angley=self.angley,
            anglez=self.anglez,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            interpolation=self.interpolation,
            fill_color=self.fill_color,
        )


class RandomPerspective(BaseRandomTransform):

    """Random perspective transformation of the image keeping center invariant
        Args:
            fov(float): range of wide angle = 90+-fov
            anglex (sequence or float or int): Range of degrees rote around X axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            angley (sequence or float or int): Range of degrees rote around Y axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            anglez (sequence or float or int): Range of degrees rote around Z axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            translate (tuple, optional): tuple of maximum absolute fraction for horizontal
                and vertical translations. For example translate=(a, b), then horizontal shift
                is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
                randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
            scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
                randomly sampled from the range a <= scale <= b. Will keep original scale by default.
            shear (sequence or float or int): Range of degrees for shear rote around axis to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to desactivate rotations.
            interpolation ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
            fill_color (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
        """

    def __init__(
        self,
        fov=0,
        anglex=0,
        angley=0,
        anglez=0,
        translate=(0, 0),
        scale=(1, 1),
        shear=0,
        interpolation="BILINEAR",
        fill_color=(0, 0, 0),
        prob=0.5,
    ):
        assert all([isinstance(anglex, (tuple, list)) or anglex >= 0,
                    isinstance(angley, (tuple, list)) or angley >= 0,
                    isinstance(anglez, (tuple, list)) or anglez >= 0,
                    isinstance(shear, (tuple, list)) or shear >= 0]), \
            "All angles must be positive or tuple or list"
        assert 80 >= fov >= 0, "fov should be in (0, 80)"
        self.fov    = fov
        self.anglex = (-anglex, anglex) if isinstance(anglex, numbers.Number) else anglex
        self.angley = (-angley, angley) if isinstance(angley, numbers.Number) else angley
        self.anglez = (-anglez, anglez) if isinstance(anglez, numbers.Number) else anglez
        self.shear  = (-shear, shear) if isinstance(shear, numbers.Number) else shear

        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "translate should be a list or tuple and it must be of length 2."
        assert all([0.0 <= i <= 1.0 for i in translate]), "translation values should be between 0 and 1"
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            assert all([s > 0 for s in scale]), "scale values should be positive"
        self.scale = scale

        self.interpolation  = interpolation
        self.fill_color = fill_color
        self.prob      = prob

    @staticmethod
    def get_params(
        fov_range,
        anglex_ranges,
        angley_ranges,
        anglez_ranges,
        translate,
        scale_ranges,
        shear_ranges,
        img_size
    ):
        fov = 90 + random.uniform(-fov_range, fov_range)
        anglex = random.uniform(anglex_ranges[0], anglex_ranges[1])
        angley = random.uniform(angley_ranges[0], angley_ranges[1])
        anglez = random.uniform(anglez_ranges[0], anglez_ranges[1])
        shear = random.uniform(shear_ranges[0], shear_ranges[1])

        max_dx = translate[0] * img_size[1]
        max_dy = translate[1] * img_size[0]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))

        scale = (random.uniform(1 / scale_ranges[0], scale_ranges[0]),
                 random.uniform(1 / scale_ranges[1], scale_ranges[1]))

        return fov, anglex, angley, anglez, translate, scale, shear

    def image_transform(self, image):
        ret = self.get_params(
            self.fov,
            self.anglex,
            self.angley,
            self.anglez,
            self.translate,
            self.scale,
            self.shear,
            image.shape,
        )
        return perspective(
            image,
            *ret,
            interpolation=self.interpolation,
            fill_color=self.fill_color,
        )
    