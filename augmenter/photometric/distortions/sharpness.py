import cv2
import copy
import random
import numbers
import numpy as np
import tensorflow as tf

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from ..blends import blend



def sharpness(metadata, factor):
    """Implements Sharpness function from PIL using TF ops."""
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
        
    orig_image = image
    image = tf.cast(image, tf.float32)
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]) / 13.0
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding="VALID", dilations=[1, 1])
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = np.ones_like(degenerate)
    padded_mask = np.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = np.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = np.where(np.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    image = blend(result, orig_image, factor)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Sharpness(BaseTransform):
    def __init__(self, factor):
        self.factor = factor

    def image_transform(self, metadata):
        return sharpness(metadata, self.factor)


class RandomSharpness(BaseRandomTransform):
    def __init__(self, factor, prob=0.5):

        if isinstance(factor, numbers.Number) and factor < 0:
            raise ValueError("Sharpness factor should be a non-negative real number")

        self.factor = factor
        self.prob = prob

    @staticmethod
    def get_params(factor):
        sharpness_factor = 1.0

        if isinstance(factor, numbers.Number) and factor > 0:
            sharpness_factor = random.uniform(0, factor)
        else:
            if factor[0] > 0 and factor[1] > 0:  
                sharpness_factor = random.uniform(factor[0], factor[1])

        if sharpness_factor < 0:
            sharpness_factor = 1.0

        return sharpness_factor
    
    def image_transform(self, metadata):
        sharpness_factor = self.get_params(self.factor)
        return sharpness(metadata, sharpness_factor)
    