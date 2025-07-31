import cv2
import copy
import math
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata


def erasing(
    metadata,
    min_area=0.02,
    max_area=1/3,
    min_aspect=0.3,
    max_aspect=None,
    min_count=1,
    max_count=1,
    fill_color=0,
    mode="constant",
):

    def _get_pixels(img, size, mode, fill_color):
        if mode == "random":
            fill_color = [random.randint(0, 255) for _ in range(img.shape[-1])]
        return np.full(size, fill_value=fill_color)

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
        
    erased_regions = []
    max_aspect = max_aspect or 1 / min_aspect
    img = image.copy()
    height, width, _ = img.shape
    area = height * width
    count = min_count if min_count == max_count else random.randint(min_count, max_count)
    log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
    
    for _ in range(count):
        for attempt in range(10):
            target_area = random.uniform(min_area, max_area) * area / count
            aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if h < img.shape[0] and w < img.shape[1]:
                top = random.randint(0, img.shape[0] - h)
                left = random.randint(0, img.shape[1] - w)
                img[top:top + h, left:left + w, :] = _get_pixels(img, (h, w, img.shape[-1]), mode, fill_color)
                erased_regions.append((left, top, left + w, top + h))
                break

    if metadata_check:
        clone_data["image"] = img
        
        # if binary_mask is not None:
        #     for (x1, y1, x2, y2) in erased_regions:
        #         binary_mask[y1:y2, x1:x2] = 0

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
                    box_area = (xmax - xmin) * (ymax - ymin)
                    erased_area = 0
    
                    for ex1, ey1, ex2, ey2 in erased_regions:
                        ix1 = max(xmin, ex1)
                        iy1 = max(ymin, ey1)
                        ix2 = min(xmax, ex2)
                        iy2 = min(ymax, ey2)
                        xw = max(0, ix2 - ix1)
                        xh = max(0, iy2 - iy1)
                        erased_area += xw * xh

                    ratio_erased = erased_area / box_area
                    if ratio_erased > 0.6:
                        boxes[i] = [0, 0, 0, 0, -1]
                        
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
            
            for i, points in enumerate(landmarks):
                for j, point in enumerate(points):
                    x, y, vis = point
                    if vis < 1:
                        continue
                    for (x1, y1, x2, y2) in erased_regions:
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            point[2] = 0
                            break
                            
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        return img


class Erasing(BaseTransform):

    def __init__(
        self,
        min_area=0.02,
        max_area=1/3,
        min_aspect=0.3,
        max_aspect=None,
        min_count=1,
        max_count=1,
        fill_color=0,
        mode="constant",
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_count = min_count
        self.max_count = max_count
        self.fill_color = fill_color
        self.mode = mode

    def image_transform(self, metadata):
        return erasing(
            metadata,
            min_area=self.min_area,
            max_area=self.max_area,
            min_aspect=self.min_aspect,
            max_aspect=self.max_aspect,
            min_count=self.min_count,
            max_count=self.max_count,
            fill_color=self.fill_color,
            mode=self.mode,
        )


class RandomErasing(BaseRandomTransform):

    def __init__(
        self,
        min_area=0.02,
        max_area=1/3,
        min_aspect=0.3,
        max_aspect=None,
        min_count=1,
        max_count=1,
        fill_color=0,
        mode="constant",
        prob=0.5,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_count = min_count
        self.max_count = max_count
        self.fill_color = fill_color
        self.mode = mode
        self.prob = prob

    def image_transform(self, metadata):
        return erasing(
            metadata,
            min_area=self.min_area,
            max_area=self.max_area,
            min_aspect=self.min_aspect,
            max_aspect=self.max_aspect,
            min_count=self.min_count,
            max_count=self.max_count,
            fill_color=self.fill_color,
            mode=self.mode,
        )
    