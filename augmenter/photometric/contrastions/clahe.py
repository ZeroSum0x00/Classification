import cv2
import copy
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata


def clahe(metadata, clip_limit=2.0, tile_grid_size=(8, 8)):
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
        
    assert (image.dtype == np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class CLAHE(BaseTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def image_transform(self, metadata):
        return clahe(metadata, self.clip_limit, self.tile_grid_size)


class RandomCLAHE(BaseRandomTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), prob=0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.prob = prob

    def image_transform(self, metadata):
        return clahe(metadata, self.clip_limit, self.tile_grid_size)
