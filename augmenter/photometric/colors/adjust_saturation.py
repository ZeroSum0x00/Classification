import cv2
import copy
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def adjust_saturation(metadata, saturation_factor):
    if isinstance(metadata, dict):
        metadata_check = True
        clone_data = copy.deepcopy(metadata)
        _, image, _, _, _, _ = extract_metadata(clone_data)
    elif isinstance(metadata, np.ndarray):
        metadata_check = False
        image = copy.deepcopy(metadata)
    else:
        raise ValueError("Input must be either a dictionary (metadata) or a NumPy array (image).")

    if image is None:
        return metadata
        
    img = image.astype(np.float32)
    degenerate = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    img = (1 - saturation_factor) * degenerate + saturation_factor * img
    img = img.clip(min=0, max=255)
    image = img.astype(image.dtype)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class AdjustSaturation(BaseTransform):
    def __init__(self, saturation_factor=1):
        self.saturation_factor = saturation_factor

    def image_transform(self, metadata):
        return adjust_saturation(metadata, self.saturation_factor)


class RandomAdjustSaturation(BaseRandomTransform):
    def __init__(self, saturation_range=1, prob=0.5):
        assert (isinstance(saturation_range, (int, float)) and saturation_range > 0) or (isinstance(saturation_range, collections.Iterable) and len(saturation_range) == 2)
        self.saturation_range = saturation_range
        self.prob = prob

    @staticmethod
    def get_params(saturation):
        saturation_factor = 1.0
        
        if isinstance(saturation, numbers.Number) and saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        elif isinstance(saturation, (tuple, list)):
            if saturation[0] > 0 and saturation[1] > 0:
                saturation_factor = random.uniform(saturation[0], saturation[1])
        
        if saturation_factor < 0:
            saturation_factor = 1.0

        return saturation_factor

    def image_transform(self, metadata):
        saturation_factor = self.get_params(self.saturation_range)
        return adjust_saturation(metadata, saturation_factor)
    