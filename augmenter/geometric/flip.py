import cv2
import copy
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.bbox_processing import coordinates_converter



def flip(metadata, mode="horizontal"):
    assert mode in ["horizontal", "vertical", "synthetic"], f"Invalid mode: {mode}"
    
    if isinstance(metadata, dict):
        metadata_check = True
        clone_data = copy.deepcopy(metadata)
        algorithm, image_data, auxi_image_data, masks_data, bbox_data, landmark_data = extract_metadata(clone_data)
        focus_image = get_focus_image_from_metadata(clone_data)
    elif isinstance(metadata, np.ndarray):
        metadata_check = False
        image_data = focus_image = copy.deepcopy(metadata)
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

    do_hflip = do_vflip = False
    if mode == "horizontal":
        do_hflip = True
    elif mode == "vertical":
        do_vflip = True
    elif mode == "synthetic":
        do_hflip = random.choice([True, False])
        do_vflip = random.choice([True, False])

    if metadata_check:
        if image_data is not None:
            img = image_data
            if do_hflip:
                img = cv2.flip(img, 1)

            if do_vflip:
                img = cv2.flip(img, 0)

            clone_data["image"] = img

        for key, array in auxi_image_data.items():
            if not isinstance(array, np.ndarray):
                continue
                
            if do_hflip:
                array = cv2.flip(array, 1)
                
            if do_vflip:
                array = cv2.flip(array, 0)
                
            clone_data["auxiliary_images"][key] = array
    
        for key, array in masks_data.items():
            if not isinstance(array, dict):
                continue

            if isinstance(array, np.ndarray):
                if do_hflip:
                    array = cv2.flip(array, 1)
                    
                if do_vflip:
                    array = cv2.flip(array, 0)
                    
                clone_data["masks"][key] = array
            elif isinstance(array, (list, tuple)):
                flipped = []
                for v in array:
                    if do_hflip:
                        v = cv2.flip(v, 1)
                    if do_vflip:
                        v = cv2.flip(v, 0)
                    flipped.append(v)
                clone_data["masks"][key] = np.array(flipped)

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
    
                if do_hflip:
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                    
                if do_vflip:
                    boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        
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
    
            if do_hflip:
                landmarks[:, :, 0] = width - 1 - landmarks[:, :, 0]
                
            if do_vflip:
                landmarks[:, :, 1] = height - 1 - landmarks[:, :, 1]
                
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        if do_hflip:
            image_data = cv2.flip(image_data, 1)

        if do_vflip:
            image_data = cv2.flip(image_data, 0)

        return image_data


class Flip(BaseTransform):

    """Flip transformation the given image.

    Args:
        mode ({horizontal, vertical, synthetic}): A flip mode.
    """

    def __init__(self, mode="horizontal"):
        self.mode = mode

    def image_transform(self, metadata):
        return flip(metadata, mode=self.mode)


class RandomFlip(BaseRandomTransform):

    """Random flip transformation the given image randomly with a given probability.

    Args:
        mode ({horizontal, vertical, synthetic}): A flip mode.
        prob (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, mode="horizontal", prob=0.5):
        self.mode = mode
        self.prob = prob

    def image_transform(self, metadata):
        return flip(metadata, mode=self.mode)


class HorizontalFlip(Flip):

    def __init__(self):
        super().__init__(mode="horizontal")


class RandomHorizontalFlip(RandomFlip):

    """Horizontally flip the given image randomly with a given probability.

    Args:
        prob (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, prob=0.5):
        super().__init__(mode="horizontal", prob=prob)
        

class VerticalFlip(Flip):

    def __init__(self):
        super().__init__(mode="vertical")

        
class RandomVerticalFlip(RandomFlip):

    """Vertically flip the given image randomly with a given probability.

    Args:
        prob (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, prob=0.5):
        super().__init__(mode="vertical", prob=prob)
    