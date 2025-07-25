import cv2
import types
import numpy as np
from utils.constants import epsilon
from utils.constants import INTER_MODE



class Normalizer:
    def __init__(
        self,
        norm_type="divide",
        target_size=(224, 224, 3),
        mean=None,
        std=None,
        interpolation="BILINEAR",
    ):
        self.norm_type = norm_type
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        
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
        
    def __call__(self, image, *args, **kargs):
        target_h, target_w = self.target_size[:2]
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=INTER_MODE[self.interpolation])

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if isinstance(self.norm_type, str):
            if self.norm_type == "divide":
                return self._divide(image)
            elif self.norm_type == "sub_divide":
                return self._sub_divide(image)
            else:
                return self._basic(image)
        elif isinstance(self.norm_type, types.FunctionType):
            return self._func_calc(image, self.norm_type)
        else:
            raise ValueError("Invalid norm_type")
        