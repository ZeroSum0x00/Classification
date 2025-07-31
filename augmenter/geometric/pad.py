import cv2
import copy
import numbers
import numpy as np
import collections.abc as collections

from augmenter.base_transform import BaseTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.bbox_processing import coordinates_converter
from utils.auxiliary_processing import is_numpy_image



PAD_MOD = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_DEFAULT,
    "symmetric": cv2.BORDER_REFLECT
}


def pad(
    metadata,
    padding,
    fill_color=(0, 0, 0),
    padding_mode="constant",
):

    def _pad_array(image):
        if not is_numpy_image(image):
            raise TypeError(f"Expected numpy image, got {type(image)}")
            
        if isinstance(fill_color, numbers.Number):
            fc = (fill_color,) * (image.shape[2] if image.ndim == 3 else 1)
        else:
            fc = fill_color
            
        return cv2.copyMakeBorder(
            image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=PAD_MOD[padding_mode],
            value=fc,
        )
        
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    elif isinstance(padding, (list, tuple)) and len(padding) == 4:
        pad_left, pad_top, pad_right, pad_bottom = padding
    else:
        raise ValueError("padding must be int, 2-tuple, or 4-tuple")

    assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

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

    if metadata_check:
        if image_data is not None:
            clone_data["image"] = _pad_array(image_data)

        for key, array in auxi_image_data.items():
            if not isinstance(array, np.ndarray):
                continue
                
            clone_data["auxiliary_images"][key] = _pad_array(array)
            
        for key, array in masks_data.items():
            if not isinstance(array, dict):
                continue

            if isinstance(array, np.ndarray):
                if array.ndim == 3 and array.shape[2] == 2 and key == "uv_mask":
                    for i in range(2):
                        array[..., i] = _pad_array(array[..., i])
                    clone_data["masks"][key] = array
                else:
                    clone_data["masks"][key] = _pad_array(array)
            elif isinstance(array, (list, np.ndarray)) and key in ["semantic_mask", "instance_mask"]:
                clone_data["masks"][key] = np.array([_pad_array(m) for m in array])
    
        offset_x, offset_y = pad_left, pad_top
    
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
    
                boxes[:, [0, 2]] += offset_x
                boxes[:, [1, 3]] += offset_y

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
            
            landmarks[:, :, 0] += offset_x
            landmarks[:, :, 1] += offset_y
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        image_data = _pad_array(image_data)
        return image_data


class Pad(BaseTransform):

    """Pad the given image on all sides with the given "pad" value.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill_color: Pixel fill_color value for constant fill_color. Default is 0. If a tuple of
            length 3, it is used to fill_color R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            constant: pads with a constant value, this value is specified with fill_color
            edge: pads with the last value at the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill_color=0, padding_mode="constant"):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill_color, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill_color = fill_color
        self.padding_mode = padding_mode

    def image_transform(self, metadata):
        return pad(metadata, self.padding, self.fill_color, self.padding_mode)
    