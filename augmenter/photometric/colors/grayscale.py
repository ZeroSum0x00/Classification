import cv2
import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def to_grayscale(metadata, out_channels=1):
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
        
    if out_channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif out_channels == 3:
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError("num_output_channels should be either 1 or 3")
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image
    

class Grayscale(BaseTransform):
    def __init__(self, out_channels=1):
        self.out_channels = out_channels

    def image_transform(self, metadata):
        return to_grayscale(metadata, self.out_channels)


class RandomGrayscale(BaseRandomTransform):
    def __init__(self, out_channels=3, prob=0.5):
        self.out_channels = out_channels
        self.prob = prob

    def image_transform(self, metadata):
        return to_grayscale(metadata, self.out_channels)
    