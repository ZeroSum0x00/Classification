import cv2
import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def jpeg_noise(metadata, quality=0.1):
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

    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(100 * quality)])
    image = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class JpegNoise(BaseTransform):
    def __init__(self, quality=0.1):
        self.quality = quality

    def image_transform(self, metadata):
        return jpeg_noise(metadata, self.quality)


class RandomJpegNoise(BaseRandomTransform):
    def __init__(self, quality=0.1, prob=0.5):
        self.quality = quality
        self.prob = prob

    def image_transform(self, metadata):
        return jpeg_noise(metadata, self.quality)
