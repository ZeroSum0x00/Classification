import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def solarize_add(metadata, add_value, threshold=128):
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
        
    idx = image < threshold
    image[idx] = np.minimum(image[idx] + add_value, 255)

    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image
    

class SolarizeAdd(BaseTransform):
    def __init__(self, add_value, threshold=128):
        self.add_value = add_value
        self.threshold = threshold

    def image_transform(self, metadata):
        return solarize_add(
            metadata,
            add_value=self.add_value,
            threshold=self.threshold
        )


class RandomSolarizeAdd(BaseRandomTransform):
    def __init__(self, add_value, threshold=128, prob=0.5):
        self.add_value = add_value
        self.threshold = threshold
        self.prob = prob

    def image_transform(self, metadata):
        return solarize_add(
            metadata,
            add_value=self.add_value,
            threshold=self.threshold
        )
    