import copy
import types

import cv2
import numpy as np
import tensorflow as tf

from utils.constants import epsilon
from augmenter import resize


def _normalize_backend(backend):
    value = (backend or "np").lower()
    if value in {"np", "numpy"}:
        return "np"
    if value in {"tf", "tensorflow"}:
        return "tf"
    raise ValueError(
        "backend must be one of 'np', 'numpy', 'tf', or 'tensorflow'. "
        f"Got: {backend}"
    )


class Normalizer:
    def __init__(
        self,
        norm_type="divide",
        target_size=(224, 224, 3),
        mean=None,
        std=None,
        interpolation="BILINEAR",
        backend="np",
    ):
        self.norm_type = norm_type
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.backend = _normalize_backend(backend)

    def _resize_np(self, image):
        return resize(
            image,
            size=self.target_size[:2],
            keep_aspect_ratio=False,
            interpolation=self.interpolation,
        )

    def _resize_tf(self, image):
        if self.target_size is None or len(self.target_size) < 2:
            return image

        method = tf.image.ResizeMethod.BILINEAR
        if isinstance(self.interpolation, str) and self.interpolation.upper() == "NEAREST":
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

        return tf.image.resize(image, self.target_size[:2], method=method)

    def _apply_mean_std_np(self, image):
        if self.mean is not None:
            for i in range(image.shape[-1]):
                if isinstance(self.mean, (float, int)):
                    image[..., i] -= self.mean
                else:
                    image[..., i] -= self.mean[i]

        if self.std is not None:
            for i in range(image.shape[-1]):
                if isinstance(self.std, (float, int)):
                    image[..., i] /= (self.std + epsilon)
                else:
                    image[..., i] /= (self.std[i] + epsilon)
        return image

    def _apply_mean_std_tf(self, image):
        if self.mean is not None:
            mean = tf.constant(self.mean, dtype=tf.float32)
            if mean.shape.ndims == 0:
                image = image - mean
            else:
                image = image - tf.reshape(mean, [1, 1, -1])

        if self.std is not None:
            std = tf.constant(self.std, dtype=tf.float32)
            if std.shape.ndims == 0:
                image = image / (std + epsilon)
            else:
                image = image / (tf.reshape(std, [1, 1, -1]) + epsilon)
        return image

    def _divide_np(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.clip(image, 0, 1)
        return self._apply_mean_std_np(image)

    def _sub_divide_np(self, image):
        image = image.astype(np.float32)
        image = image / 127.5 - 1.0
        image = np.clip(image, -1, 1)
        return self._apply_mean_std_np(image)

    def _basic_np(self, image):
        image = image.astype(np.uint8)
        image = np.clip(image, 0, 255)
        return self._apply_mean_std_np(image)

    def _divide_tf(self, image):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return self._apply_mean_std_tf(image)

    def _sub_divide_tf(self, image):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, tf.float32)
        image = image * 2.0 - 1.0
        image = tf.clip_by_value(image, -1.0, 1.0)
        return self._apply_mean_std_tf(image)

    def _basic_tf(self, image):
        if image.dtype != tf.uint8:
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = tf.cast(image, tf.uint8)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return self._apply_mean_std_tf(image)

    def _normalize_np(self, image):
        if isinstance(self.norm_type, str):
            if self.norm_type == "divide":
                return self._divide_np(image)
            if self.norm_type == "sub_divide":
                return self._sub_divide_np(image)
            return self._basic_np(image)
        if isinstance(self.norm_type, types.FunctionType):
            return self.norm_type(image)
        raise ValueError("Invalid norm_type")

    def _normalize_tf(self, image):
        if isinstance(self.norm_type, str):
            if self.norm_type == "divide":
                return self._divide_tf(image)
            if self.norm_type == "sub_divide":
                return self._sub_divide_tf(image)
            return self._basic_tf(image)
        if isinstance(self.norm_type, types.FunctionType):
            return self.norm_type(image)
        raise ValueError("Invalid norm_type")

    def normalize(self, image):
        if self.backend == "tf":
            return self._normalize_tf(image)
        return self._normalize_np(image)

    def __call__(self, metadata, *args, **kwargs):
        if isinstance(metadata, dict):
            metadata_check = True
            clone_data = copy.deepcopy(metadata)
            image = clone_data.get("image")
        elif isinstance(metadata, np.ndarray):
            metadata_check = False
            image = metadata
        elif tf.is_tensor(metadata):
            metadata_check = False
            image = metadata
        else:
            raise ValueError(
                "Input must be a metadata dict, NumPy array, or TensorFlow tensor."
            )

        if self.backend == "tf":
            if len(image.shape) == 2:
                image = tf.expand_dims(image, axis=-1)
            image = self._resize_tf(image)
            image = self.normalize(image)
        else:
            if metadata_check:
                clone_data = resize(
                    clone_data,
                    size=self.target_size[:2],
                    keep_aspect_ratio=False,
                    interpolation=self.interpolation,
                )
                image = clone_data.get("image")
            else:
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                image = self._resize_np(image)
            image = self.normalize(image)

        if metadata_check:
            clone_data["image"] = image
            return clone_data
        return image
