import math
import random

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import get_focus_image_from_metadata
from .resize import resize



class ResizeKeepRatio(BaseTransform):
    def __init__(self, size, interpolation="BILINEAR"):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
        self.interpolation = interpolation

    def get_params(self, orin_size):
        in_h, in_w = orin_size[:2]
        target_h, target_w = self.size

        scale = max(target_h / in_h, target_w / in_w)
        new_h = int(round(in_h * scale))
        new_w = int(round(in_w * scale))

        return (new_h, new_w)

    def image_transform(self, metadata):
        image = get_focus_image_from_metadata(metadata)
        new_h, new_w = self.get_params(image.shape)
        return resize(metadata, (new_h, new_w), self.interpolation)


class RandomResizeKeepRatio(BaseRandomTransform):
    def __init__(
        self,
        size,
        longest=0.,
        random_scale_prob=0.5,
        random_scale_range=(0.8, 1.2),
        random_scale_area=True,
        random_aspect_prob=0.5,
        random_aspect_range=(0.75, 1.3333),
        interpolation="BILINEAR",
        prob=0.5
    ):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
        self.longest = longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_scale_area = random_scale_area
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range
        self.interpolation = interpolation
        self.prob = prob

    def get_params(self, orin_size):
        in_h, in_w = orin_size[:2]
        target_h, target_w = self.size

        ratio_h = in_h / target_h
        ratio_w = in_w / target_w
        ratio = max(ratio_h, ratio_w) * self.longest + min(ratio_h, ratio_w) * (1. - self.longest)

        if self.random_scale_prob > 0 and random.random() < self.random_scale_prob:
            ratio_factor = random.uniform(*self.random_scale_range)
            if self.random_scale_area:
                ratio_factor = 1. / math.sqrt(ratio_factor)
        else:
            ratio_factor = 1.0

        aspect_factor = 1.0
        if self.random_aspect_prob > 0 and random.random() < self.random_aspect_prob:
            log_aspect = (math.log(self.random_aspect_range[0]), math.log(self.random_aspect_range[1]))
            aspect = math.exp(random.uniform(*log_aspect))
            aspect_factor = math.sqrt(aspect)

        scale_h = ratio_factor / aspect_factor
        scale_w = ratio_factor * aspect_factor

        new_h = int(round(in_h * scale_h / ratio))
        new_w = int(round(in_w * scale_w / ratio))

        new_h = max(new_h, target_h)
        new_w = max(new_w, target_w)

        return (new_h, new_w)

    def image_transform(self, metadata):
        image = get_focus_image_from_metadata(metadata)
        new_h, new_w = self.get_params(image.shape)        
        return resize(metadata, (new_h, new_w), self.interpolation)
        