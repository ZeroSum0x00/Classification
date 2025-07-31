import cv2
import copy
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def adjust_hue(metadata, hue_factor):
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError("Hue factor is not in [-0.5, 0.5].".format(hue_factor))

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
        
    img = image.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(hue_factor * 255)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    image = img.astype(image.dtype)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class AdjustHue(BaseTransform):
    def __init__(self, hue_factor=0):
        self.hue_factor = hue_factor

    def image_transform(self, metadata):
        return adjust_hue(metadata, self.hue_factor)


class RandomAdjustHue(BaseRandomTransform):
    def __init__(self, hue_range=0, prob=0.5):
        assert (isinstance(hue_range, (int, float)) and hue_range > 0) or (isinstance(hue_range, collections.Iterable) and len(hue_range) == 2)
        self.hue_range = hue_range
        self.prob = prob

    @staticmethod
    def get_params(hue):
        hue_factor = 0.0

        if isinstance(hue, numbers.Number) and hue > 0:
            hue_factor = random.uniform(-hue, hue)
        elif isinstance(hue, (tuple, list)):
            hue_factor = random.uniform(hue[0], hue[1])

        if not(-0.5 <= hue_factor <= 0.5):
            hue_factor = 0.0
            
        return hue_factor

    def image_transform(self, metadata):
        hue_factor = self.get_params(self.hue_range)
        return adjust_hue(metadata, hue_factor)
    