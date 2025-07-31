import copy
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def linear_gradient(metadata, orientation="horizontal", edge_brightness=(.1, .3)):
    assert isinstance(edge_brightness, tuple) and len(edge_brightness) == 2, \
        "Argument edge_brightness should be a tuple with size 2."
    assert 0. < edge_brightness[0] < 1. and 0. < edge_brightness[1] < 1., \
        "Values of an edge_brightness argument should be in the [0, 1] range."
    assert orientation in ["horizontal", "vertical"], "Unknown orientation value."

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

    color1 = int(edge_brightness[0] * 255)
    color2 = int(edge_brightness[1] * 255)
    reverse = bool(random.getrandbits(1))

    image = np.int16(image)
    dim = image.shape[1] if orientation == "horizontal" else image.shape[0]
    for i in range(dim):
        coeff = i / float(dim)
        if reverse:
            coeff = 1. - coeff
        diff = int((color2 - color1) * coeff)
        if orientation == "horizontal":
            image[:, i, 0:3] = np.where(
                image[:, i, 0:3] + color1 + diff < 255,
                image[:, i, 0:3] + color1 + diff,
                255,
            )
        else:
            image[i, :, 0:3] = np.where(
                image[i, :, 0:3] + color1 + diff < 255,
                image[i, :, 0:3] + color1 + diff,
                255,
            )
    image = image.astype(np.uint8)

    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class LinearGradient(BaseTransform):
    def __init__(
        self,
        orientation="horizontal",
        edge_brightness=(.1, .3),
    ):
        self.orientation = orientation
        self.edge_brightness = edge_brightness

    def image_transform(self, metadata):
        return linear_gradient(metadata, self.orientation, self.edge_brightness)


class RandomLinearGradient(BaseRandomTransform):
    def __init__(
        self,
        orientation="horizontal",
        edge_brightness=(.1, .3),
        prob=0.5,
    ):
        self.orientation = orientation
        self.edge_brightness = edge_brightness
        self.prob = prob

    def image_transform(self, metadata):
        return linear_gradient(metadata, self.orientation, self.edge_brightness)
    