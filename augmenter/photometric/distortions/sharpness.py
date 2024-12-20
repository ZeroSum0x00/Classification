import cv2
import random
import numbers
import numpy as np
import tensorflow as tf

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image
from ..blends import blend


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
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
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return blend(result, orig_image, factor)


class Sharpness(BaseTransform):
    def __init__(self, factor):
        self.factor = factor

    def image_transform(self, image):
        return sharpness(image, self.factor)


class RandomSharpness(BaseRandomTransform):
    def __init__(self, factor, prob=0.5):

        if isinstance(factor, numbers.Number) and factor < 0:
            raise ValueError('Sharpness factor should be a non-negative real number')

        self.factor = factor
        self.prob   = prob

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
    
    def image_transform(self, image):
        sharpness_factor = self.get_params(self.factor)
        return sharpness(image, sharpness_factor)