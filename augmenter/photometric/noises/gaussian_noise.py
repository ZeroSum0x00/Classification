import copy
import random
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def gaussian_noise(metadata, mean, std):
    assert isinstance(mean, numbers.Number) and mean >= 0, "mean should be a positive value"
    assert isinstance(std, numbers.Number) and std >= 0, "std should be a positive value"

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
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = np.clip((1 + gauss) * image.astype(np.float32), 0, 255)
    image = noisy.astype(imgtype)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class GaussianNoise(BaseTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def image_transform(self, metadata):
        return gaussian_noise(metadata, self.mean, self.std)


class RandomGaussianNoise(BaseRandomTransform):
    def __init__(self, mean_range=0, std_range=0.1, prob=0.5):
        assert isinstance(mean_range, (int, float)) or (isinstance(mean_range, collections.Iterable) and len(mean_range) == 2)
        assert isinstance(std_range, (int, float)) or (isinstance(std_range, collections.Iterable) and len(std_range) == 2)
        self.mean_range = mean_range
        self.std_range = std_range
        self.prob = prob

    @staticmethod
    def get_params(mean, std):
        if isinstance(mean, numbers.Number):
            mean_factor = random.uniform(0, mean)
        else:
            mean_factor = random.uniform(mean[0], mean[1])

        if isinstance(std, numbers.Number):
            std_factor = random.uniform(0, std)
        else:
            std_factor = random.uniform(std[0], std[1])
        return mean_factor, std_factor

    def image_transform(self, metadata):
        ret = self.get_params(self.mean_range, self.std_range)
        return gaussian_noise(metadata, *ret)
    