import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata


def solarize(metadata, threshold=128):
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
        
    idx = image >= threshold
    image[idx] = 255 - image[idx]

    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Solarize(BaseTransform):
    def __init__(self, threshold=128):
        self.threshold = threshold

    def image_transform(self, metadata):
        return solarize(metadata, self.threshold)


class RandomSolarize(BaseRandomTransform):
    def __init__(self, threshold=128, prob=0.5):
        self.threshold = threshold
        self.prob = prob

    def image_transform(self, metadata):
        return solarize(metadata, self.threshold)
    