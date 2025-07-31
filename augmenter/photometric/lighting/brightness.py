import copy
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from ..blends import blend



def brightness(metadata, factor):
    if factor < 0:
        raise ValueError("Brightness should be a non-negative real number")
        
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

    degenerate = np.zeros_like(image)
    image = blend(degenerate, image, factor)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Brightness(BaseTransform):
    def __init__(self, factor):
        self.factor = factor

    def image_transform(self, metadata):
        return brightness(metadata, self.factor)


class RandomBrightness(BaseRandomTransform):
    def __init__(self, factor, prob=0.5):

        if isinstance(factor, numbers.Number) and factor < 0:
            raise ValueError("Brightness factor should be a non-negative real number")

        self.factor = factor
        self.prob = prob

    @staticmethod
    def get_params(factor):
        brightness_factor = 1.0

        if isinstance(factor, numbers.Number) and factor > 0:
            brightness_factor = random.uniform(0, factor)
        else:
            if factor[0] > 0 and factor[1] > 0:  
                brightness_factor = random.uniform(factor[0], factor[1])

        if brightness_factor < 0:
            brightness_factor = 1.0

        return brightness_factor
    
    def image_transform(self, metadata):
        brightness_factor = self.get_params(self.factor)
        return brightness(metadata, brightness_factor)
    