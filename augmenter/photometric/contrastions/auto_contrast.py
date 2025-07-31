import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.constants import epsilon



def auto_contrast(metadata):
    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = np.min(image).astype(np.float32)
        hi = np.max(image).astype(np.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo + epsilon)
            offset = -lo * scale
            im = im.astype(np.float32) * scale + offset
            im = np.clip(im, 0.0, 255.0)
            return im.astype(np.uint8)

        result = np.where(hi > lo, scale_values(image), image)
        return result

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
        
    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class AutoContrast(BaseTransform):

    def image_transform(self, metadata):
        return auto_contrast(metadata)


class RandomAutoContrast(BaseRandomTransform):

    def image_transform(self, metadata):
        return auto_contrast(metadata)
    