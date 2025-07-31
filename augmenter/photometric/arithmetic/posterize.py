import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def posterize(metadata, bits):
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
        
    if bits >= 8:
        return metadata

    img_dtype = image.dtype
    is_float = np.issubdtype(img_dtype, np.floating)
    
    if is_float:
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Float images must be normalized in the [0, 1] range.")
        image_uint8 = (image * 255).astype(np.uint8)
    elif img_dtype == np.uint8:
        image_uint8 = image
    else:
        raise TypeError(f"Unsupported dtype: {img_dtype}. Only float32 and uint8 are supported.")

    mask = ~(2 ** (8 - bits) - 1)
    image_uint8 = image_uint8 & mask

    if is_float:
        image = image_uint8.astype(np.float32) / 255.0
    else:
        image = image_uint8.astype(np.uint8)

    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image
        

class Posterize(BaseTransform):
    def __init__(self, bits):
        self.bits = bits

    def image_transform(self, metadata):
        return posterize(metadata, bits=self.bits)


class RandomPosterize(BaseRandomTransform):
    def __init__(self, bits, prob=0.5):
        self.bits = bits
        self.prob = prob

    def image_transform(self, metadata):
        return posterize(metadata, bits=self.bits)
    