import cv2
import copy
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.bbox_processing import coordinates_converter



def shear(metadata, factor_x=0.0, factor_y=0.0):
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

    M = np.array([
        [1, factor_x, 0],
        [factor_y, 1, 0]
    ], dtype=np.float32)

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
                clone_data["masks"][key] = cv2.warpAffine(array, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    
            elif isinstance(array, (list, tuple)):
                clone_data["masks"][key] = np.array([
                    cv2.warpAffine(v, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0) for v in array
                ])

        if bbox_data.get("value") is not None:
            boxes = copy.deepcopy(bbox_data.get("value"))
            coord = bbox_data.get("coord", "corners")
            max_box = bbox_data.get("max_box", 100)
            
            if algorithm.lower() == "od":
                if not isinstance(boxes, np.ndarray) or boxes.ndim != 2 or boxes.shape[1] != 5:
                    raise ValueError(f"Expected boxes to be Nx5 numpy array. Got shape: {boxes.shape}")
                
                out_boxes = np.zeros((max_box, 5))
                out_boxes[:, -1] = -1
                np.random.shuffle(boxes)
                
                if coord == "centroids":
                    boxes = coordinates_converter(boxes, conversion="centroids2corners")
        
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                points = np.stack([
                    np.stack([x1, y1], axis=1),
                    np.stack([x2, y1], axis=1),
                    np.stack([x1, y2], axis=1),
                    np.stack([x2, y2], axis=1)
                ], axis=1)
        
                ones = np.ones((points.shape[0], 4, 1), dtype=np.float32)
                homo = np.concatenate([points, ones], axis=2)
        
                transformed = np.matmul(homo, M.T)
                min_xy = transformed.min(axis=1)
                max_xy = transformed.max(axis=1)
                boxes[:, 0:2] = min_xy
                boxes[:, 2:4] = max_xy
    
                if coord == "centroids":
                    boxes = coordinates_converter(boxes, conversion="corners2centroids")
                    
                if len(boxes) > max_box: 
                    boxes = boxes[:max_box]
                    
                out_boxes[:len(boxes)] = boxes
                clone_data["bounding_box"]["value"] = out_boxes

        if landmark_data.get("value") is not None:
            landmarks = copy.deepcopy(landmark_data.get("value"))
            
            if landmarks.ndim == 2 and landmarks.shape[1] == landmark.get("num_points", 3)*3:
                landmarks = landmarks.reshape(-1, m, 3)
            elif landmarks.ndim != 3 or landmarks.shape[2] != 3:
                raise ValueError(f"Landmark shape invalid: {landmarks.shape}")
    
            N, K, _ = landmarks.shape
    
            coords = landmarks[:, :, :2]
            ones = np.ones((N, K, 1), dtype=np.float32)
            homo = np.concatenate([coords, ones], axis=2)
    
            transformed = np.matmul(homo, M.T)
            landmarks[:, :, :2] = transformed
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        return cv2.warpAffine(image_data, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class Shear(BaseTransform):
    def __init__(self, factor_x, factor_y):
        self.factor_x = factor_x
        self.factor_y = factor_y

    def image_transform(self, metadata):
        return shear(metadata, factor_x=self.factor_x, factor_y=self.factor_y)


class RandomShear(BaseRandomTransform):
    def __init__(self, factor_x, factor_y, prob=0.5):
        self.factor_x = factor_x
        self.factor_y = factor_y
        self.prob = prob

    @staticmethod
    def get_params(factor):
        shear_factor = 0.0

        if isinstance(factor, numbers.Number) and factor > 0:
            shear_factor = random.uniform(-factor, factor)
        elif isinstance(factor, (tuple, list)):
            shear_factor = random.uniform(factor[0], factor[1])

        return shear_factor

    def image_transform(self, metadata):
        factor_x = self.get_params(self.factor_x)
        factor_y = self.get_params(self.factor_y)
        return shear(metadata, factor_x=factor_x, factor_y=factor_y)


class ShearX(Shear):
    def __init__(self, factor_x):
        super().__init__(factor_x=factor_x, factor_y=0)


class RandomShearX(RandomShear):
    def __init__(self, factor_x, prob=0.5):
        super().__init__(factor_x=factor_x, factor_y=0, prob=prob)

    def image_transform(self, metadata):
        factor_x = self.get_params(self.factor_x)
        return shear(metadata, factor_x=factor_x, factor_y=0)


class ShearY(Shear):
    def __init__(self, factor_y):
        super().__init__(factor_x=0, factor_y=factor_y)


class RandomShearY(RandomShear):
    def __init__(self, factor_y, prob=0.5):
        super().__init__(factor_x=0, factor_y=factor_y, prob=prob)

    def image_transform(self, metadata):
        factor_y = self.get_params(self.factor_y)
        return shear(metadata, factor_x=0, factor_y=factor_y)


