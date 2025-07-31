import cv2
import copy
import types
import numpy as np
from utils.constants import epsilon
from utils.constants import INTER_MODE
from augmenter import resize



class Normalizer:
    def __init__(
        self,
        norm_type="divide",
        target_size=(224, 224, 3),
        mean=None,
        std=None
    ):
        self.norm_type = norm_type
        self.target_size = target_size
        self.mean = mean
        self.std = std
        
    def __get_standard_deviation(self, img):
        if self.mean is not None:
            for i in range(img.shape[-1]):
                if isinstance(self.mean, float) or isinstance(self.mean, int):
                    img[..., i] -= self.mean
                else:
                    img[..., i] -= self.mean[i]

        if self.std is not None:
            for i in range(img.shape[-1]):
                if isinstance(self.std, float) or isinstance(self.std, int):
                    img[..., i] /= (self.std + epsilon)
                else:
                    img[..., i] /= (self.std[i] + epsilon)
        return img
    
    def _sub_divide(self, image):
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = np.clip(image, -1, 1)
        image = self.__get_standard_deviation(image)
        return image

    def _divide(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.clip(image, 0, 1)
        image = self.__get_standard_deviation(image)
        return image

    def _basic(self, image):
        image = image.astype(np.uint8)
        image = np.clip(image, 0, 255)
        image = self.__get_standard_deviation(image)
        return image

    def _func_calc(self, image, func):
        image = func(image)
        return image
        
    def __call__(self, metadata, *args, **kargs):
        metadata = resize(
            metadata,
            size=self.target_size[:2],
            keep_aspect_ratio=False
        )

        if isinstance(metadata, dict):
            metadata_check = True
            clone_data = copy.deepcopy(metadata)
            image = clone_data.get("image")
        elif isinstance(metadata, np.ndarray):
            metadata_check = False
            image = metadata
        else:
            raise ValueError("Input must be either a dictionary (metadata) or a NumPy array (image).")
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if isinstance(self.norm_type, str):
            if self.norm_type == "divide":
                image = self._divide(image)
            elif self.norm_type == "sub_divide":
                image = self._sub_divide(image)
            else:
                image = self._basic(image)
        elif isinstance(self.norm_type, types.FunctionType):
            image = self._func_calc(image)
        else:
            raise ValueError("Invalid norm_type")

        if metadata_check:
            clone_data["image"] = image
            return clone_data
        else:
            return image