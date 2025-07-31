import cv2
import copy
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.bbox_processing import coordinates_converter
from utils.auxiliary_processing import is_numpy_image



def translate(metadata, dx, dy):
    if isinstance(metadata, dict):
        metadata_check = True
        clone_data = copy.deepcopy(metadata)
        algorithm, image_data, auxi_image_data, masks_data, bbox_data, landmark_data = extract_metadata(clone_data)
        focus_image = get_focus_image_from_metadata(clone_data)
    elif isinstance(metadata, np.ndarray):
        image_data = focus_image = copy.deepcopy(metadata)
        metadata_check = False
    else:
        raise ValueError("Input must be either a dictionary (metadata) or a NumPy array (image).")
    
    if focus_image.ndim == 2:
        height, width = focus_image.shape
        gray_scale = True
    elif focus_image.ndim == 3:
        height, width, _ = focus_image.shape
        gray_scale = False
    else:
        raise ValueError(f"Unsupported image shape: {focus_image.shape}")

    dx = int(dx * width) if abs(dx) <= 1 else int(dx)
    dy = int(dy * height) if abs(dy) <= 1 else int(dy)

    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

    if metadata_check:
        if image_data is not None:
            img = image_data
            img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            clone_data["image"] = img
            
        for key, array in auxi_image_data.items():
            if not isinstance(array, np.ndarray):
                continue
                
            clone_data["auxiliary_images"][key] = cv2.warpAffine(array, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        for key, array in masks_data.items():
            if not isinstance(array, dict):
                continue

            if isinstance(array, np.ndarray):
                array = cv2.warpAffine(array, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                clone_data["masks"][key] = array
                
            elif isinstance(array, (list, tuple)):
                translated = []
                for v in array:
                    v = cv2.warpAffine(v, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    translated.append(v)
                    
                clone_data["masks"][key] = np.array(translated)
                
        if bbox_data.get("value") is not None:
            boxes = copy.deepcopy(bbox_data.get("value"))
            coord = bbox_data.get("coord", "corners")
            max_box = bbox_data.get("max_box", 100)
            clip_out_range = bbox_data.get("clip_out_range", True)
            
            if algorithm.lower() == "od":
                if not isinstance(boxes, np.ndarray) or boxes.ndim != 2 or boxes.shape[1] != 5:
                    raise ValueError(f"Expected boxes to be Nx5 numpy array. Got shape: {boxes.shape}")
                
                out_boxes = np.zeros((max_box, 5))
                out_boxes[:, -1] = -1
                np.random.shuffle(boxes)
                
                if coord == "centroids":
                    boxes = coordinates_converter(boxes, conversion="centroids2corners")
    
                boxes[:, [0, 2]] += dx
                boxes[:, [1, 3]] += dy
    
                if clip_out_range:
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
                    
                if coord == "centroids":
                    boxes = coordinates_converter(boxes, conversion="corners2centroids")
                    
                if len(boxes) > max_box: 
                    boxes = boxes[:max_box]
                    
                out_boxes[:len(boxes)] = boxes
                clone_data["bounding_box"]["value"] = out_boxes

        if landmark_data.get("value") is not None:
            landmarks = copy.deepcopy(landmark_data.get("value"))
            clip_out_range = landmark_data.get("clip_out_range", False)
            
            if landmarks.ndim == 2 and landmarks.shape[1] == landmark.get("num_points", 3)*3:
                landmarks = landmarks.reshape(-1, m, 3)
            elif landmarks.ndim != 3 or landmarks.shape[2] != 3:
                raise ValueError(f"Landmark shape invalid: {landmarks.shape}")
    
            landmarks[:, :, 0] += dx
            landmarks[:, :, 1] += dy
            
            if clip_out_range:
                landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, width - 1)
                landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, height - 1)
            
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        return cv2.warpAffine(image_data, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class Translate(BaseTransform):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def image_transform(self, metadata):
        return translate(metadata, dx=self.dx, dy=self.dy)


class RandomTranslate(BaseRandomTransform):
    def __init__(self, dx, dy, prob=0.5):
        self.dx = dx
        self.dy = dy
        self.prob = prob

    @staticmethod
    def get_params(factor):
        translate_factor = 0.0

        if isinstance(factor, numbers.Number) and factor > 0:
            translate_factor = random.uniform(-factor, factor)
        elif isinstance(factor, (tuple, list)):
            translate_factor = random.uniform(factor[0], factor[1])

        return translate_factor

    def image_transform(self, metadata):
        dx = self.get_params(self.dx)
        dy = self.get_params(self.dy)
        return translate(metadata, dx=dx, dy=dy)

        
class TranslateX(Translate):
    def __init__(self, dx):
        super().__init__(dx=dx, dy=0)


class RandomTranslateX(RandomTranslate):
    def __init__(self, dx, prob=0.5):
        super().__init__(dx=dx, dy=0, prob=prob)

    def image_transform(self, metadata):
        dx = self.get_params(self.dx)
        return translate(metadata, dx=dx, dy=0)


class TranslateY(Translate):
    def __init__(self, dy):
        super().__init__(dx=0, dy=dy)


class RandomTranslateY(RandomTranslate):
    def __init__(self, dy, prob=0.5):
        super().__init__(dx=0, dy=dy, prob=prob)

    def image_transform(self, metadata):
        translate_factor = self.get_params(self.dy)
        return translate(metadata, dx=0, dy=translate_factor)
