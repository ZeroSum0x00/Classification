import cv2
import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def erosion_or_dilation(metadata, kernel_size=5, reversed=False):
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
        
    kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    er, dil = cv2.erode, cv2.dilate
    
    if reversed:
        er, dil = dil, er

    image[:, :, 0] = er(image[:, :, 0], kernel, iterations=1)
    image[:, :, 1] = er(image[:, :, 1], kernel, iterations=1)
    image[:, :, 2] = er(image[:, :, 2], kernel, iterations=1)

    if image.shape[2] > 3:
        image[:, :, 3] = dil(image[:, :, 3], kernel, iterations=1)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Erosion(BaseTransform):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def image_transform(self, metadata):
        return erosion_or_dilation(metadata, self.kernel_size, reversed=False)


class RandomErosion(BaseRandomTransform):
    def __init__(self, kernel_size=5, prob=0.5):
        self.kernel_size = kernel_size
        self.prob = prob

    def image_transform(self, metadata):
        return erosion_or_dilation(metadata, self.kernel_size, reversed=False)


class Dilation(BaseTransform):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def image_transform(self, metadata):
        return erosion_or_dilation(metadata, self.kernel_size, reversed=True)


class RandomDilation(BaseRandomTransform):
    def __init__(self, kernel_size=5, prob=0.5):
        self.kernel_size = kernel_size
        self.prob = prob

    def image_transform(self, metadata):
        return erosion_or_dilation(metadata, self.kernel_size, reversed=True)
    