import cv2
import copy
import random
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.bbox_processing import coordinates_converter
from utils.constants import INTER_MODE



def resize(
    metadata,
    size=None,
    keep_aspect_ratio=False,
    interpolation="BILINEAR"
):
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
    elif focus_image.ndim == 3:
        height, width, _ = focus_image.shape
    else:
        raise ValueError(f"Unsupported image shape: {focus_image.shape}")

    aspect = width / height
    if isinstance(size, int):
        if keep_aspect_ratio:
            if (width <= height and width == size) or (height <= width and height == size):
                return metadata
                
            if width < height:
                ow = size
                oh = int(size * aspect)
            else:
                oh = size
                ow = int(size * aspect)
        else:
            oh = ow = size
    else:
        if keep_aspect_ratio:
            target_h, target_w = size
            
            if target_w / target_h > aspect:
                oh = target_h
                ow = int(aspect * target_h)
            else:
                ow = target_w
                oh = int(target_w / aspect)
        else:
            oh, ow = size

    scale_x = ow / width
    scale_y = oh / height

    if metadata_check:
        if image_data is not None:
            clone_data["image"] = cv2.resize(image_data, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation])
            
        for key, array in auxi_image_data.items():
            if not isinstance(array, np.ndarray):
                continue
                
            clone_data["auxiliary_images"][key] = cv2.resize(array, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation])
        
        for key, array in masks_data.items():
            if not isinstance(array, dict):
                continue

            if isinstance(array, np.ndarray):
                if array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 1):
                    if key in ["binary_mask", "edge_mask", "part_mask", "instance_mask"]:
                        interp = INTER_MODE["NEAREST"]
                    elif key in ["saliency_mask", "depth_mask"]:
                        interp = INTER_MODE["LINEAR"]
                    else:
                        interp = INTER_MODE["NEAREST"]
                    clone_data["masks"][key] = cv2.resize(array, dsize=(int(ow), int(oh)), interpolation=interp)
        
                # Resize 3D UV map: (H, W, 2)
                elif array.ndim == 3 and array.shape[2] == 2 and key == "uv_mask":
                    resized_uv = np.zeros((int(oh), int(ow), 2), dtype=value.dtype)
                    for i in range(2):
                        resized_uv[..., i] = cv2.resize(array[..., i], dsize=(int(ow), int(oh)), interpolation=INTER_MODE["LINEAR"])
                    clone_data["masks"][key] = resized_uv
        
            # Resize list of instance/semantic masks
            elif isinstance(array, (list, np.ndarray)) and key in ["semantic_mask", "instance_mask"]:
                resized_list = [cv2.resize(m, dsize=(int(ow), int(oh)), interpolation=INTER_MODE["NEAREST"]) for m in array]
                clone_data["masks"][key] = np.array(resized_list)
    
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
                    
                boxes[:, [0, 2]] = np.round(boxes[:, [0, 2]] * scale_x, decimals=0)
                boxes[:, [1, 3]] = np.round(boxes[:, [1, 3]] * scale_y, decimals=0)
    
                if clip_out_range:
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow - 1)
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh - 1)
                    
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
    
            landmarks[:, :, 0] = np.round(landmarks[:, :, 0] * scale_x, decimals=0)
            landmarks[:, :, 1] = np.round(landmarks[:, :, 1] * scale_y, decimals=0)
    
            if clip_out_range:
                landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, ow - 1)
                landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, oh - 1)
            
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        return cv2.resize(image_data, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation])


class Resize(BaseTransform):

    """Resize the input image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is ``BILINEAR``
    """

    def __init__(self, size, keep_aspect_ratio=True):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.keep_aspect_ratio = keep_aspect_ratio

    def image_transform(self, metadata):
        return resize(
            metadata,
            size=self.size,
            keep_aspect_ratio=self.keep_aspect_ratio
        )
    