import copy
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def channel_shuffle(metadata):
    if isinstance(metadata, dict):
        metadata_check = True
        clone_data = copy.deepcopy(metadata)
        _, image, _, _, _, _ = extract_metadata(clone_data)
    elif isinstance(metadata, np.ndarray):
        metadata_check = False
        image = copy.deepcopy(metadata)
    else:
        raise ValueError("Input must be either a dictionary (metadata) or a NumPy array (image).")

    assert image.shape[2] in [3, 4]

    if image is None:
        return metadata
        
    ch_arr = [0, 1, 2]
    random.shuffle(ch_arr)
    image = image[..., ch_arr]

    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image

        
class ChannelShuffle(BaseTransform):

    def image_transform(self, metadata):
        return channel_shuffle(metadata)


class RandomChannelShuffle(BaseRandomTransform):

    def image_transform(self, metadata):
        return channel_shuffle(metadata)
