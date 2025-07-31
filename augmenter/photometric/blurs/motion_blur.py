import cv2
import copy
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def motion_blur(metadata, ksize_norm=0.1):
    def point_near_line(px, py, x1, y1, x2, y2, threshold=5):
        # Check if a point (px, py) is within `threshold` pixels from line segment (x1, y1)-(x2, y2)
        if x1 == x2 and y1 == y2:
            return np.hypot(px - x1, py - y1) <= threshold
        line_mag = np.hypot(x2 - x1, y2 - y1)
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
        u = np.clip(u, 0, 1)
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        dist = np.hypot(px - ix, py - iy)
        return dist <= threshold
    
    if isinstance(metadata, dict):
        metadata_check = True
        clone_data = copy.deepcopy(metadata)
        algorithm, image, auxi_image_data, masks_data, bbox_data, landmark_data = extract_metadata(clone_data)
    elif isinstance(metadata, np.ndarray):
        image = copy.deepcopy(metadata)
        metadata_check = False
    else:
        raise ValueError("Input must be either a dictionary (metadata) or a NumPy array (image).")
        
    if image is None:
        return metadata

    if image.ndim == 2:
        height, width = image.shape
    elif image.ndim == 3:
        height, width, _ = image.shape
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
        
    k_size = int(min(height, width) * ksize_norm)
    k_size = k_size + 1 if k_size % 2 == 0 else k_size
    if k_size <= 2:
        return metadata

    x1, x2 = random.randint(0, k_size - 1), random.randint(0, k_size - 1)
    y1, y2 = random.randint(0, k_size - 1), random.randint(0, k_size - 1)

    kernel_mtx = np.zeros((k_size, k_size), dtype=np.float32)
    cv2.line(kernel_mtx, (x1, y1), (x2, y2), 1, thickness=1)
    kernel_mtx /= np.sum(kernel_mtx)
    
    image = cv2.filter2D(image, -1, kernel_mtx)

    if metadata_check:
        clone_data["image"] = img
        
        for key, array in masks_data.items():
            if not isinstance(array, dict):
                continue

            if array.ndim == 2:
                blurred_mask = cv2.filter2D(array, -1, kernel_mtx)
                blurred_mask = (blurred_mask > 127).astype(array.dtype) * 255
            else:
                blurred_mask = np.zeros_like(array)
                for i in range(array.shape[-1]):
                    blurred = cv2.filter2D(array[..., i], -1, kernel_mtx)
                    blurred_mask[..., i] = (blurred > 127).astype(array.dtype) * 255
            clone_data["masks"][key] = blurred_mask
        
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
    
                new_boxes = []
                for i, box in enumerate(boxes):
                    xmin, ymin, xmax, ymax, label = box
                    cx = (xmin + xmax) // 2
                    cy = (ymin + ymax) // 2
                    
                    if not point_near_line(cx, cy, x1, y1, x2, y2, threshold=7):
                        new_boxes.append([xmin, ymin, xmax, ymax, label])
    
                boxes = np.array(new_boxes)
    
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
    
            for group in landmarks:
                for point in group:
                    x, y, vis = point
                    if vis >= 1 and point_near_line(x, y, x1, y1, x2, y2, threshold=5):
                        point[2] = 0  # set visibility to 0 (blurred out)

            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        return image


class MotionBlur(BaseTransform):
    def __init__(self, ksize_norm=0.1):
        self.ksize_norm = ksize_norm

    def image_transform(self, metadata):
        return motion_blur(metadata, self.ksize_norm)


class RandomMotionBlur(BaseRandomTransform):
    def __init__(self, ksize_norm=0.1, prob=0.5):
        self.ksize_norm = ksize_norm
        self.prob = prob

    def image_transform(self, metadata):
        return motion_blur(metadata, self.ksize_norm)
 