import cv2
import copy
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def adjust_contrast(metadata, contrast_factor):
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
        
    image = image.astype(np.float32)
    mean = round(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean())
    image = (1 - contrast_factor) * mean + contrast_factor * image
    image = image.clip(min=0, max=255).astype(np.uint8)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image
    

class AdjustContrast(BaseTransform):
    def __init__(self, contrast_factor=1):
        self.contrast_factor = contrast_factor

    def image_transform(self, metadata):
        return adjust_contrast(metadata, self.contrast_factor)


class RandomAdjustContrast(BaseRandomTransform):
    def __init__(self, contrast_range=1, prob=0.5):
        assert (isinstance(contrast_range, (int, float)) and contrast_range > 0) or (isinstance(contrast_range, collections.Iterable) and len(contrast_range) == 2)
        self.contrast_range = contrast_range
        self.prob = prob

    @staticmethod
    def get_params(contrast):
        contrast_factor = 1.0

        if isinstance(contrast, numbers.Number):
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        elif isinstance(contrast, (tuple, list)):
            if contrast[0] > 0 and contrast[1] > 0:
                contrast_factor = random.uniform(contrast[0], contrast[1])

        if contrast_factor < 0:
            contrast_factor = 1.0

        return contrast_factor

    def image_transform(self, metadata):
        contrast_factor = self.get_params(self.contrast_range)
        return adjust_contrast(metadata, contrast_factor)
    