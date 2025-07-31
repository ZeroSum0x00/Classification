import cv2
import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def pixelize(metadata, ratio=.2):
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

    height, width = image.shape[:2]
    tmp_w = max(1, int(width * ratio))
    tmp_h = max(1, int(height * ratio))

    image = cv2.resize(image, (tmp_w, tmp_h), interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Pixelize(BaseTransform):
    def __init__(self, ratio=.2):
        self.ratio = ratio

    def image_transform(self, metadata):
        return pixelize(metadata, self.ratio)

class RandomPixelize(BaseRandomTransform):
    def __init__(self, ratio=.2, prob=0.5):
        self.ratio = ratio
        self.prob = prob

    def image_transform(self, metadata):
        return pixelize(metadata, self.ratio)
