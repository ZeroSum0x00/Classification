import cv2
import copy
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def adjust_brightness(metadata, brightness_factor):
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
        
    image = image.astype(np.float32) * brightness_factor
    image = image.clip(min=0, max=255).astype(np.uint8)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class AdjustBrightness(BaseTransform):
    def __init__(self, brightness_factor=1):
        self.brightness_factor = brightness_factor

    def image_transform(self, metadata):
        return adjust_brightness(metadata, self.brightness_factor)


class RandomAdjustBrightness(BaseRandomTransform):
    def __init__(self, brightness_range=1, prob=0.5):
        assert (isinstance(brightness_range, (int, float)) and brightness_range > 0) or (isinstance(brightness_range, collections.Iterable) and len(brightness_range) == 2)
        self.brightness_range = brightness_range
        self.prob = prob

    @staticmethod
    def get_params(brightness):
        brightness_factor = 1.0

        if isinstance(brightness, numbers.Number):
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            if brightness[0] > 0 and brightness[1] > 0:
                brightness_factor = random.uniform(brightness[0], brightness[1])

        if brightness_factor < 0:
            brightness_factor = 1.0

        return brightness_factor
    
    def image_transform(self, metadata):
        brightness_factor = self.get_params(self.brightness_range)
        return adjust_brightness(metadata, brightness_factor)
    