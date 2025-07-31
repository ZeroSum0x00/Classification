import cv2
import copy
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from ..pad import pad
from utils.auxiliary_processing import is_numpy_image



def pad_if_needed(image, ymin, xmin, ymax, xmax):
    h, w = image.shape[:2]

    pad_top = -min(0, ymin)
    pad_bottom = max(ymax - h, 0)
    pad_left = -min(0, xmin)
    pad_right = max(xmax - w, 0)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        image = cv2.copyMakeBorder(
            image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0] * image.shape[2] if image.ndim == 3 else 0,
        )
    return image
    
    
def crop(metadata, top, left, height, width):
    assert height > 0 and width > 0, f"height={height} and width={width} must be > 0"
    
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
        ih, iw = focus_image.shape
        gray_scale = True
    elif focus_image.ndim == 3:
        ih, iw, _ = focus_image.shape
        gray_scale = False
    else:
        raise ValueError(f"Unsupported image shape: {focus_image.shape}")

    ymin, xmin = round(top), round(left)
    ymax, xmax = ymin + height, xmin + width

    if metadata_check:
        if image_data is not None:
            img = image_data
            img = pad_if_needed(img, ymin, xmin, ymax, xmax)
            img = img[ymin:ymax, xmin:xmax, ...]
            clone_data["image"] = img

        for key, array in auxi_image_data.items():
            if not isinstance(array, np.ndarray):
                continue
                
            array = pad_if_needed(array, ymin, xmin, ymax, xmax)
            array = array[ymin:ymax, xmin:xmax, ...]
            clone_data["auxiliary_images"][key] = array
            
        for key, array in masks_data.items():
            if not isinstance(array, dict):
                continue
        
            if isinstance(array, np.ndarray):
                mask = pad_if_needed(array, ymin, xmin, ymax, xmax)
                mask = mask[ymin:ymax, xmin:xmax, ...]
                clone_data["masks"][key] = mask
    
            elif isinstance(array, (list, tuple)) and key in ["semantic_mask", "instance_mask"]:
                cropped = []
                for v in array:
                    v = pad_if_needed(v, ymin, xmin, ymax, xmax)
                    v = v[ymin:ymax, xmin:xmax]
                    cropped.append(v)
                clone_data["masks"][key] = np.array(cropped)

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

                boxes[:, [0, 2]] -= xmin
                boxes[:, [1, 3]] -= ymin
        
                x1 = np.clip(boxes[:, 0], 0, width - 1)
                y1 = np.clip(boxes[:, 1], 0, height - 1)
                x2 = np.clip(boxes[:, 2], 0, width - 1)
                y2 = np.clip(boxes[:, 3], 0, height - 1)
        
                keep = (x2 > x1) & (y2 > y1)
                boxes = boxes[keep]
                x1 = x1[keep]
                y1 = y1[keep]
                x2 = x2[keep]
                y2 = y2[keep]
                cls = boxes[:, 4]
        
                boxes = np.stack([x1, y1, x2, y2, cls], axis=1)
        
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
            
            landmarks[:, :, 0] -= xmin
            landmarks[:, :, 1] -= ymin
            
            inside = (landmarks[:, :, 0] >= 0) & (landmarks[:, :, 0] < width) & \
                     (landmarks[:, :, 1] >= 0) & (landmarks[:, :, 1] < height)
    
            landmarks[:, :, 2] = landmarks[:, :, 2] * inside.astype(np.float32)
            
            if clip_out_range:
                landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, width - 1)
                landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, height - 1)
    
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:    
        image_data = pad_if_needed(image_data, ymin, xmin, ymax, xmax)
        image_data = image_data[ymin:ymax, xmin:xmax, ...]
        return image_data


class Crop(BaseTransform):

    """Crop the given image to desired size.

    Args:
        top: Upper pixel coordinate.
        left: Left pixel coordinate.
        height: Height of the cropped image.
        width: Width of the cropped image.
    """

    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def image_transform(self, metadata):
        return crop(metadata, self.top, self.left, self.height, self.width)


class RandomCrop(BaseRandomTransform):

    """Crop the given image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill_color (number or tuple or dict, optional): Pixel fill_color value used when 
            the padding_mode is constant. Default is 0. If a tuple of length 3,
            it is used to fill_color R, G, B channels respectively.
        padding_mode (str, optional): Type of padding. Should be: constant,
        edge, reflect or symmetric. Default is constant.
    """

    def __init__(
        self,
        size,
        padding=0,
        pad_if_needed=False,
        fill_color=0,
        padding_mode="constant",
        prob=0.5
    ):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
            
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill_color = fill_color
        self.padding_mode = padding_mode
        self.prob = prob

    @staticmethod
    def get_params(image, size):
        h, w = image.shape[:2]
        th, tw = size
        
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, max(h - th, 0))
        j = random.randint(0, max(w - tw, 0))
        return i, j, th, tw

    def image_transform(self, metadata):        
        if self.padding > 0:
            metadata = pad(
                metadata,
                padding=(self.padding, self.padding, self.padding, self.padding),
                fill_color=self.fill_color,
                padding_mode=self.padding_mode,
            )
            
        focus_image = get_focus_image_from_metadata(metadata)
        height, width = focus_image.shape[:2]

        pad_left = pad_right = pad_top = pad_bottom = 0

        if self.pad_if_needed:
            pad_left = max((self.size[1] - width) // 2, 0)
            pad_right = max(self.size[1] - width - pad_left, 0)
            pad_top = max((self.size[0] - height) // 2, 0)
            pad_bottom = max(self.size[0] - height - pad_top, 0)

        if height < self.size[0] or width < self.size[1]:
            pad_top = max(self.size[0] - height, 0) // 2
            pad_bottom = max(self.size[0] - height - pad_top, 0)
            pad_left = max(self.size[1] - width, 0) // 2
            pad_right = max(self.size[1] - width - pad_left, 0)
            
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            metadata = pad(
                metadata,
                padding=(pad_left, pad_top, pad_right, pad_bottom),
                fill_color=self.fill_color,
                padding_mode=self.padding_mode,
            )
                
        focus_image = get_focus_image_from_metadata(metadata)
        top, left, crop_h, crop_w = self.get_params(focus_image, self.size)
        return crop(metadata, top, left, crop_h, crop_w)
        