import cv2
import copy
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from ..blends import blend


def color(metadata, color_factor):
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
        
    degenerate = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    degenerate = cv2.cvtColor(degenerate, cv2.COLOR_GRAY2RGB)
    image = blend(degenerate, image, color_factor)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Color(BaseTransform):
    def __init__(self, color_factor=1):
        self.color_factor = color_factor

    def image_transform(self, metadata):
        return color(metadata, self.color_factor)


class RandomColor(BaseRandomTransform):
    def __init__(self, color_range=1, prob=0.5):
        assert (isinstance(color_range, (int, float)) and color_range > 0) or (isinstance(color_range, collections.Iterable) and len(color_range) == 2)
        self.color_range = color_range
        self.prob = prob

    @staticmethod
    def get_params(color):
        color_factor = 0.0

        if isinstance(color, numbers.Number) and color > 0:
            color_factor = random.uniform(-color, color)
        elif isinstance(color, (tuple, list)):
            color_factor = random.uniform(color[0], color[1])

        if not(0. < color_factor < 1):
            color_factor = 0.0
            
        return color_factor

    def image_transform(self, metadata):
        color_factor = self.get_params(self.color_range)
        return color(metadata, color_factor)
    