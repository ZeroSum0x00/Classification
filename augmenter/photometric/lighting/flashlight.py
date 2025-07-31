import cv2
import copy
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image, compute_iou
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def flashlight(metadata, radius=0.5, alpha=0.8, bg_darkness=100):
    def decrease_brightness(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    if isinstance(metadata, dict):
        metadata_check = True
        clone_data = copy.deepcopy(metadata)
        algorithm, image_data, auxi_image_data, masks_data, bbox_data, landmark_data = extract_metadata(clone_data)
        image = image_data.get("value")
    elif isinstance(metadata, np.ndarray):
        image = copy.deepcopy(metadata)
        metadata_check = False
    else:
        raise ValueError("Input must be either a dictionary (metadata) or a NumPy array (image).")
        
    if image is None:
        return metadata

    height, width = image.shape[:2]

    min_wh = min(width, height)
    max_wh = max(width, height)
    radius = int(random.randint(min_wh, max_wh) * radius)

    pos_x = random.randint(int(1 / 4 * width), int(3 / 4 * width))
    pos_y = random.randint(int(1 / 4 * height), int(3 / 4 * height))

    k = random.uniform(1.5, 5.)
    blur_kernel_size = (int(radius / k), int(radius / k))
    
    circle_mask = np.zeros((height, width), dtype=np.uint8)
    circle_mask = cv2.circle(circle_mask, (pos_x, pos_y), radius, 255, -1)
    circle_mask = cv2.blur(circle_mask, blur_kernel_size)

    darkened_image = decrease_brightness(image, value=bg_darkness)

    # create white circle on black background
    torchlight = np.zeros((height, width, 3), np.uint8)
    cv2.circle(torchlight, (pos_x, pos_y), radius, (255, 255, 255), -1)

    blurred_torchlight = cv2.blur(torchlight, blur_kernel_size)
    image = cv2.addWeighted(darkened_image, 1.0, blurred_torchlight, alpha, 0.0)

    if metadata_check:
        clone_data["image"] = image
        
        if bbox_data.get("value") is not None:
            boxes = copy.deepcopy(bbox_data.get("value"))
            coord = bbox_data.get("coord", "corners")
            max_box = bbox_data.get("max_box", 100)
            clip_out_range = bbox_data.get("clip_out_range", True)
    
            x1_flare = max(pos_x - radius, 0)
            y1_flare = max(pos_y - radius, 0)
            x2_flare = min(pos_x + radius, width - 1)
            y2_flare = min(pos_y + radius, height - 1)
            flare_box = (x1_flare, y1_flare, x2_flare, y2_flare)
            
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
                    box_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.rectangle(box_mask, (xmin, ymin), (xmax, ymax), 255, -1)
    
                    iou = compute_iou((xmin, ymin, xmax, ymax), flare_box)
                    
                    if iou < 0.13:
                        new_boxes.append(box)
                    else:
                        new_boxes.append([0, 0, 0, 0, 1])
                        
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
            
            for i, points in enumerate(landmarks):
                for j, point in enumerate(points):
                    x, y, vis = point
                    
                    if x < 0 or x >= width:
                        continue
    
                    if y < 0 or y >= height:
                        continue
    
                    if vis == 0:
                        continue
    
                    if circle_mask[y, x] == 0:
                        pass
                    else:
                        landmarks[i, j, -1] = 0
                        
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        return image


class Flashlight(BaseTransform):
    def __init__(self, radius=0.5, alpha=0.8, bg_darkness=100):
        self.radius = radius
        self.alpha = alpha
        self.bg_darkness = bg_darkness

    def image_transform(self, metadata):
        return flashlight(metadata, self.radius, self.alpha, self.bg_darkness)


class RandomFlashlight(BaseRandomTransform):
    def __init__(self, radius=0.5, alpha=0.8, bg_darkness=100, prob=0.5):
        self.radius = radius
        self.alpha = alpha
        self.bg_darkness = bg_darkness
        self.prob = prob

    def image_transform(self, metadata):
        return flashlight(metadata, self.radius, self.alpha, self.bg_darkness)
    