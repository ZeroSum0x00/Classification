import copy
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def adjust_gamma(metadata, gamma, gain=1):
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")

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
    image = 255. * gain * np.power(image / 255., gamma)
    image = image.clip(min=0., max=255.).astype(np.uint8)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class AdjustGamma(BaseTransform):
    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def image_transform(self, metadata):
        return adjust_gamma(metadata, self.gamma, self.gain)


class RandomAdjustGamma(BaseRandomTransform):
    def __init__(self, gamma, gain=1, prob=0.5):

        if isinstance(gamma, numbers.Number) and gamma < 0:
            raise ValueError("Gamma should be a non-negative real number")

        self.gamma = gamma
        self.gain = gain
        self.prob = prob

    @staticmethod
    def get_params(gamma):
        gamma_factor = 1.0

        if isinstance(gamma, numbers.Number) and gamma > 0:
            gamma_factor = random.uniform(0, gamma)
        else:
            if gamma[0] > 0 and gamma[1] > 0:  
                gamma_factor = random.uniform(gamma[0], gamma[1])

        if gamma_factor < 0:
            gamma_factor = 1.0

        return gamma_factor
    
    def image_transform(self, metadata):
        gamma_factor = self.get_params(self.gamma)
        return adjust_gamma(metadata, gamma_factor, gain=self.gain)
    