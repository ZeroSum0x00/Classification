import cv2
import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata


def gaussian_blur(metadata, ksize_norm=.4, sigma=5, direction=None):
    assert direction in ("horizontal", "vertical", None)
    
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

    k_size = int(min(image.shape[:2]) * ksize_norm)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size
    if k_size <= 2:
        return metadata

    if direction == "horizontal":
        image = cv2.GaussianBlur(image, (k_size, 1), sigmaX=sigma, sigmaY=sigma)
    elif direction == "vertical":
        image = cv2.GaussianBlur(image, (1, k_size), sigmaX=sigma, sigmaY=sigma)
    else:
        image = cv2.GaussianBlur(image, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)

    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image
    

class GaussianBlur(BaseTransform):
    def __init__(self, ksize_norm=.4, sigma=5, direction=None):
        self.ksize_norm = ksize_norm
        self.sigma = sigma
        self.direction = direction

    def image_transform(self, metadata):
        return gaussian_blur(metadata, self.ksize_norm, self.sigma, self.direction)


class RandomGaussianBlur(BaseRandomTransform):
    def __init__(self, ksize_norm=.4, sigma=5, direction=None, prob=0.5):
        self.ksize_norm = ksize_norm
        self.sigma = sigma
        self.direction = direction
        self.prob = prob

    def image_transform(self, metadata):
        return gaussian_blur(metadata, self.ksize_norm, self.sigma, self.direction)
 