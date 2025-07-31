import io
import cv2
import copy
import random
import requests
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.logger import logger



def blend_random_image(metadata, ratio=0.8):
    URL = "https://picsum.photos/{}/{}/?random"

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
        
    h, w = image.shape[:2]
    try:
        r = requests.get(URL.format(w, h), allow_redirects=True)
        f = io.BytesIO(r.content)
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        random_img  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if random_img is None:
            return metadata
            
        image = cv2.addWeighted(image, ratio, random_img, 1 - ratio, 0)
        image = image.astype(np.uint8)
        
        if metadata_check:
            clone_data["image"] = image
            return clone_data
        else:
            return image
    except requests.exceptions.ConnectionError as e:
        logger.error("Unable to download image. Error: {}".format(e))
    except Exception as e:
        logger.error("Unknown error occurred '{}'".format(e))


class BlendRandomImage(BaseTransform):
    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def image_transform(self, metadata):
        return blend_random_image(metadata, self.ratio)


class RandomBlendRandomImage(BaseRandomTransform):
    def __init__(self, ratio=0.8, prob=0.5):
        self.ratio = ratio
        self.prob = prob

    def image_transform(self, metadata):
        return blend_random_image(metadata, self.ratio)
    