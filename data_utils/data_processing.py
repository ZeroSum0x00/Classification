import os
import cv2
import xml
import types
import numpy as np
from tqdm import tqdm

from utils.files import extract_zip, verify_folder, get_files, valid_image


def extract_data_folder(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar', '.tar']
    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = verify_folder(data_destination) + folder_name 

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        
        return data_destination
    else:
        return data_dir


def get_data(data_dir, classes, data_type=None, phase='train', check_data=False, load_memory=False):
    data_dir = verify_folder(data_dir) + phase + '/'

    data_extraction = []
    for idx, name in enumerate(tqdm(classes, desc=f"Load {phase} dataset")):
        files_name = get_files(f"{data_dir}{name}", extensions=['jpg', 'jpeg', 'png'])
        
        for image_name in files_name:
            if check_data:
                image_path = os.path.join(data_dir, name, image_name)
                try:
                    valid_image(image_path)
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    shape = img.shape
                except Exception as e:
                    print(f"Error: File {image_name} is can't loaded: {e}")
                    continue
                
            info_dict = {}
            info_dict['image'] = None
            info_dict['filename'] = image_name
            info_dict['label'] = name
            
            if load_memory:
                image_path = os.path.join(data_dir, name, image_name)
                img = cv2.imread(image_path)
                info_dict['image'] = img
                
            data_extraction.append(info_dict)
            
    dict_data = {
        'data_path': verify_folder(data_dir),
        'data_extractor': data_extraction
    }
    return dict_data


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
                    img[..., i] /= (self.std + 1e-20)
                else:
                    img[..., i] /= (self.std[i] + 1e-20)
        return img
    
    def _sub_divide(self, image, target_size=None, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = cv2.resize(image, (target_size[0], target_size[1]), interpolation=interpolation)
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = self.__get_standard_deviation(image)
        image = np.clip(image, -1, 1)
        return image

    def _divide(self, image, target_size=None, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = cv2.resize(image, (target_size[0], target_size[1]), interpolation=interpolation)
        image = image.astype(np.float32)
        image = image / 255.0
        image = self.__get_standard_deviation(image)
        image = np.clip(image, 0, 1)
        return image

    def _basic(self, image, target_size, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = cv2.resize(image, (target_size[0], target_size[1]), interpolation=interpolation)
        image = image.astype(np.uint8)
        image = self.__get_standard_deviation(image)
        image = np.clip(image, 0, 255)
        return image

    def _func_calc(self, image, func, target_size, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image = cv2.resize(image, (target_size[0], target_size[1]), interpolation=interpolation)
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
