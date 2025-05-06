import math
import random

from augmenter.base_transform import BaseTransform
from .resize import resize


class ResizeKeepRatio(BaseTransform):
    def __init__(
        self,
        size,
        longest=0.,
        interpolation="BILINEAR",
        random_scale_prob=0.,
        random_scale_range=(0.85, 1.05),
        random_scale_area=False,
        random_aspect_prob=0.,
        random_aspect_range=(0.9, 1.11),
    ):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_scale_area = random_scale_area
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
        orin_size,
        target_size,
        longest,
        random_scale_prob=0.,
        random_scale_range=(1.0, 1.33),
        random_scale_area=False,
        random_aspect_prob=0.,
        random_aspect_range=(0.9, 1.11),
    ):
        target_h, target_w = target_size
        ratio_h = orin_size[0] / target_h
        ratio_w = orin_size[1] / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)

        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            if random_scale_area:
                # make ratio factor equivalent to RRC area crop where < 1.0 = area zoom,
                # otherwise like affine scale where < 1.0 = linear zoom out
                ratio_factor = 1. / math.sqrt(ratio_factor)
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)

        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            log_aspect = (math.log(random_aspect_range[0]), math.log(random_aspect_range[1]))
            aspect_factor = math.exp(random.uniform(*log_aspect))
            aspect_factor = math.sqrt(aspect_factor)
            # currently applying random aspect adjustment equally to both dims,
            # could change to keep output sizes above their target where possible
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)

        size = [round(x * f / ratio) for x, f in zip(orin_size, ratio_factor)]
        return size

    def image_transform(self, image):
        orin_size = image.shape[:2]
        target_size = self.get_params(
            orin_size=orin_size,
            target_size=self.size,
            longest=self.longest,
            random_scale_prob=self.random_scale_prob,
            random_scale_range=self.random_scale_range,
            random_scale_area=self.random_scale_area,
            random_aspect_prob=self.random_aspect_prob,
            random_aspect_range=self.random_aspect_range,
        )
        return resize(image, target_size, self.interpolation)
    