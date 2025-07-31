import cv2
import copy
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from ..blends import blend



def contrast(metadata, contrast_factor):
    if contrast_factor < 0:
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

    degenerate = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist, _ = np.histogram(degenerate, bins=256, range=[0, 255])
    mean = np.sum(hist) / 256.0
    degenerate = np.ones_like(degenerate, dtype=np.float32) * mean
    degenerate = np.clip(degenerate, 0.0, 255.0)
    degenerate = degenerate.astype(np.uint8)
    degenerate = cv2.cvtColor(degenerate, cv2.COLOR_GRAY2BGR)
    image = blend(degenerate, image, contrast_factor)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Contrast(BaseTransform):
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def image_transform(self, metadata):
        return contrast(metadata, self.contrast_factor)


class RandomContrast(BaseRandomTransform):
    def __init__(self, contrast_range, prob=0.5):

        if isinstance(contrast_range, numbers.Number) and contrast_range < 0:
            raise ValueError("Contrast factor should be a non-negative real number")

        self.contrast_range = contrast_range
        self.prob = prob

    @staticmethod
    def get_params(factor):
        contrast_factor = 1.0

        if isinstance(factor, numbers.Number) and factor > 0:
            contrast_factor = random.uniform(0, factor)
        else:
            if factor[0] > 0 and factor[1] > 0:  
                contrast_factor = random.uniform(factor[0], factor[1])

        if contrast_factor < 0:
            contrast_factor = 1.0

        return contrast_factor
    
    def image_transform(self, metadata):
        contrast_factor = self.get_params(self.contrast_range)
        return contrast(metadata, contrast_factor)
    