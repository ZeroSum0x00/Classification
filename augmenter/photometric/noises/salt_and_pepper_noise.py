import copy
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def salt_and_pepper_noise(metadata, threshold=0.01):
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

    imgtype = image.dtype
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy = image.copy()
    noisy[rnd < threshold / 2] = 0.0
    noisy[rnd > 1 - threshold / 2] = 255.0
    image = noisy.astype(imgtype)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image
        

class SaltPepperNoise(BaseTransform):
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def image_transform(self, metadata):
        return salt_and_pepper_noise(metadata, self.threshold)


class RandomSaltPepperNoise(BaseRandomTransform):
    def __init__(self, threshold_range=0.1, prob=0.5):
        assert isinstance(threshold_range, (int, float)) or (isinstance(threshold_range, collections.Iterable) and len(threshold_range) == 2)
        self.threshold_range = threshold_range
        self.prob = prob

    @staticmethod
    def get_params(threshold):
        if isinstance(threshold, numbers.Number):
            threshold_factor = random.uniform(0, threshold)
        else:
            threshold_factor = random.uniform(threshold[0], threshold[1])
        return threshold_factor

    def image_transform(self, metadata):
        threshold_factor = self.get_params(self.threshold_range)
        return salt_and_pepper_noise(metadata, threshold_factor)
    