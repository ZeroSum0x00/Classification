import copy
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.constants import epsilon



def equalize(metadata):
    """Implements Equalize function from PIL using TF ops."""
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :, c].astype(np.int32)
        # Compute the histogram of the image channel.
        histo = np.histogram(im, 256, [0, 256])[0]

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = np.where(np.not_equal(histo, 0))
        nonzero_histo = np.reshape(np.take(histo, nonzero), [-1])
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (np.cumsum(histo) + (step // 2)) // (step + epsilon)
            # Shift lut, prepending with 0.
            lut = np.concatenate([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return np.clip(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = np.where(np.equal(step, 0), im, np.take(build_lut(histo, step), im))
        return result.astype(np.uint8)
        
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
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = np.stack([s1, s2, s3], 2)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Equalize(BaseTransform):

    def image_transform(self, metadata):
        return equalize(metadata)


class RandomEqualize(BaseRandomTransform):
  
    def image_transform(self, metadata):
        return equalize(metadata)
    