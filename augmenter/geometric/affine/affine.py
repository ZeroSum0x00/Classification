import cv2
import copy
import math
import random
import numbers
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.bbox_processing import coordinates_converter
from ..perspective import apply_transform
from ..resize import INTER_MODE




def affine(
    metadata,
    angle=0,
    translate=(0, 0),
    scale=1,
    shear=0,
    fill_color=(0, 0, 0),
    interpolation="BILINEAR"
):
    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"
    assert scale > 0.0, "Argument scale should be positive"
    
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

    center = (width * 0.5, height * 0.5)
    angle = math.radians(angle)
    shear = math.radians(shear)

    M00 = math.cos(angle) * scale
    M01 = -math.sin(angle + shear) * scale
    M10 = math.sin(angle) * scale
    M11 = math.cos(angle + shear) * scale
    M02 = center[0] - center[0] * M00 - center[1] * M01 + translate[0]
    M12 = center[1] - center[0] * M10 - center[1] * M11 + translate[1]
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)
    total_matrix = np.vstack([affine_matrix, [0, 0, 1]])
    
    if metadata_check:
        if image_data is not None:
            img = image_data

            if gray_scale:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
            img = cv2.warpAffine(
                img,
                affine_matrix,
                (width, height),
                flags=INTER_MODE[interpolation],
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=fill_color,
            )
            
            if gray_scale:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            clone_data["image"] = img
            
        for key, array in auxi_image_data.items():
            if not isinstance(array, np.ndarray):
                continue
            
            if gray_scale:
                array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)

            array = cv2.warpAffine(
                array,
                affine_matrix,
                (width, height),
                flags=INTER_MODE[interpolation],
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=fill_color,
            )
            
            if gray_scale:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

            clone_data["auxiliary_images"][key] = array

        for key, array in masks_data.items():
            if not isinstance(array, dict):
                continue
        
            if isinstance(array, np.ndarray):
                clone_data["masks"][key] = cv2.warpAffine(
                    array,
                    affine_matrix,
                    (width, height),
                    flags=INTER_MODE["NEAREST"],
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=fill_color,
                )
            elif isinstance(array, (list, tuple)):
                clone_data["masks"][key] = [cv2.warpAffine(
                    v,
                    affine_matrix,
                    (width, height),
                    flags=INTER_MODE["NEAREST"],
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=fill_color,
                ) for v in array]

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
    
                for i, box in enumerate(boxes):
                    xmin, ymin, xmax, ymax, label = box
                    points = np.array([
                        [xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]
                    ], dtype=np.float32)
                    transformed = apply_transform(points, total_matrix)
                    transformed_2d = transformed[:, :2]
                
                    # Tạo lại bbox từ các điểm
                    new_xmin = np.min(transformed_2d[:, 0])
                    new_ymin = np.min(transformed_2d[:, 1])
                    new_xmax = np.max(transformed_2d[:, 0])
                    new_ymax = np.max(transformed_2d[:, 1])
                    boxes[i] = [new_xmin, new_ymin, new_xmax, new_ymax, label]
    
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
                landmarks[i] = apply_transform(points, total_matrix)
    
            if clip_out_range:
                landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, width - 1)
                landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, height - 1)
    
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        if gray_scale:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
                
        image_data = cv2.warpAffine(
            image_data,
            affine_matrix,
            (width, height),
            flags=INTER_MODE[interpolation],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill_color,
        )
        
        if gray_scale:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

        return image_data


class Affine(BaseTransform):

    """Affine transformation of the image keeping center invariant

    Args:
        angle (sequence or float or int): Range of angle to select from.
            If angle is a number instead of sequence like (min, max), the range of angle
            will be (-angle, +angle). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of angle to select from.
            If angle is a number instead of sequence like (min, max), the range of angle
            will be (-angle, +angle). Will not apply shear by default
        interpolation ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
        fill_color (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(
        self,
        angle,
        translate=(0, 0),
        scale=1,
        shear=0,
        fill_color=(0, 0, 0),
    ):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill_color = fill_color

    def image_transform(self, metadata):
        return affine(
            metadata,
            self.angle,
            self.translate,
            self.scale,
            self.shear,
            self.fill_color,
        )


class RandomAffine(BaseRandomTransform):

    """Random affine transformation of the image keeping center invariant

    Args:
        angle (sequence or float or int): Range of angle to select from.
            If angle is a number instead of sequence like (min, max), the range of angle
            will be (-angle, +angle). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of angle to select from.
            If angle is a number instead of sequence like (min, max), the range of angle
            will be (-angle, +angle). Will not apply shear by default
        interpolation ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
        fill_color (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(
        self,
        angle=0,
        translate=None,
        scale=None,
        shear=None,
        fill_color=0,
        prob=0.5,
    ):
        if isinstance(angle, numbers.Number):
            if angle < 0:
                raise ValueError("If angle is a single number, it must be positive.")
            self.angle = (-angle, angle)
        else:
            assert isinstance(angle, (tuple, list)) and len(angle) == 2, \
                "angle should be a list or tuple and it must be of length 2."
            self.angle = angle

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.fill_color = fill_color
        self.prob = prob

    @staticmethod
    def get_params(angle, translate, scale_ranges, shears, img_size):
        angle = random.uniform(angle[0], angle[1])
        if translate is not None:
            max_dx = translate[0] * img_size[1]
            max_dy = translate[1] * img_size[0]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def image_transform(self, metadata):
        focus_image = get_focus_image_from_metadata(metadata)
        
        angle, translations, scale, shear = self.get_params(
            self.angle,
            self.translate,
            self.scale,
            self.shear,
            focus_image.shape,
        )
        return affine(
            metadata,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            fill_color=self.fill_color,
        )
    