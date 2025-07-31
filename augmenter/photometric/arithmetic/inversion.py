import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata


def inversion(metadata):
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
        
    image[:, :, :3] = 255 - image[:, :, :3]
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image
        

class Inversion(BaseTransform):

    def image_transform(self, metadata):
        return inversion(metadata)

class RandomInversion(BaseRandomTransform):

    def image_transform(self, metadata):
        return inversion(metadata)
