import os
import cv2
import xml
import types
import numpy as np
from tqdm import tqdm

from data_utils import ParseDirName
from utils.files import extract_zip, get_files, valid_image
from utils.constants import ALLOW_IMAGE_EXTENSIONS, epsilon


def extract_data_folder(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar', '.tar']
    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = os.path.join(data_destination, folder_name) 

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        
        return data_destination
    else:
        return data_dir


def get_data(data_dirs, classes, data_type=None, phase='train', check_data=False, load_memory=False, *args, **kwargs):

    def load_data(data_dir):
        if data_type.lower() == "dirname":
            image_file_list = [sorted(get_files(os.path.join(data_dir, cls), ALLOW_IMAGE_EXTENSIONS, cls)) for cls in classes]
            image_files = [item for sublist in image_file_list for item in sublist]
            parser = ParseDirName(data_dir, classes, load_memory, check_data=check_data, *args, **kwargs)
            return parser(image_files)

    assert data_type.lower() in ('dirname')
    data_extraction = []

    if isinstance(data_dirs, (list, tuple)):
        for data_dir in data_dirs:
            data_dir = os.path.join(data_dir, phase)
            parser = load_data(data_dir)
            data_extraction.extend(parser)
    else:
        data_dir = os.path.join(data_dirs, phase)
        parser = load_data(data_dir)
        data_extraction.extend(parser)
        
    return data_extraction


class Normalizer:
    def __init__(self, norm_type="divide", mean=None, std=None):
        self.norm_type = norm_type
        self.mean      = mean
        self.std       = std
        
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
    
    def _sub_divide(self, image, target_size=None, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = np.clip(image, -1, 1)
        image = self.__get_standard_deviation(image)
        return image

    def _divide(self, image, target_size=None, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image = image.astype(np.float32)
        image = image / 255.0
        image = np.clip(image, 0, 1)
        image = self.__get_standard_deviation(image)
        return image

    def _basic(self, image, target_size, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image = image.astype(np.uint8)
        image = np.clip(image, 0, 255)
        image = self.__get_standard_deviation(image)
        return image

    def _func_calc(self, image, func, target_size, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image = func(image)
        return image
        
    def __call__(self, input, *args, **kargs):
        if isinstance(self.norm_type, str):
            if self.norm_type == "divide":
                return self._divide(input, *args, **kargs)
            elif self.norm_type == "sub_divide":
                return self._sub_divide(input, *args, **kargs)
        elif isinstance(self.norm_type, types.FunctionType):
            return self._func_calc(input, self.norm_type, *args, **kargs)
        else:
            return self._basic(input, *args, **kargs)