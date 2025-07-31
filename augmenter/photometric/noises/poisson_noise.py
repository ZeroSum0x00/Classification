import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def poisson_noise(metadata):
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

    imgtype = image.dtype
    image = image.astype(np.float32)/255.0
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = 255 * np.clip(np.random.poisson(image.astype(np.float32) * vals) / float(vals), 0, 1)
    image = noisy.astype(imgtype)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class PoissonNoise(BaseTransform):

    def image_transform(self, metadata):
        return poisson_noise(metadata)


class RandomPoissonNoise(BaseRandomTransform):

    def image_transform(self, metadata):
        return poisson_noise(metadata)
    