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



def affine6(
    metadata,
    anglez=0,
    translate=(0, 0),
    scale=(1, 1),
    shear=0,
    fill_color=(0, 0, 0),
    interpolation="BILINEAR"
):
    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"
    
    assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
        "Argument translate should be a list or tuple of length 2"

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

    centery = height * 0.5
    centerx = width * 0.5
    
    alpha = math.radians(shear)
    beta = math.radians(anglez)

    lambda1 = scale[0]
    lambda2 = scale[1]

    tx = translate[0]
    ty = translate[1]

    sina = math.sin(alpha)
    cosa = math.cos(alpha)
    sinb = math.sin(beta)
    cosb = math.cos(beta)

    M00 = cosb * (lambda1 * cosa**2 + lambda2 * sina**2) - sinb * (lambda2 - lambda1) * sina * cosa
    M01 = - sinb * (lambda1 * sina**2 + lambda2 * cosa**2) + cosb * (lambda2 - lambda1) * sina * cosa

    M10 = sinb * (lambda1 * cosa**2 + lambda2 * sina**2) + cosb * (lambda2 - lambda1) * sina * cosa
    M11 = + cosb * (lambda1 * sina**2 + lambda2 * cosa**2) + sinb * (lambda2 - lambda1) * sina * cosa
    M02 = centerx - M00 * centerx - M01 * centery + tx
    M12 = centery - M10 * centerx - M11 * centery + ty
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


class Affine6(BaseTransform):

    """Affine transformation of the image keeping center invariant

    Args:
        anglez (sequence or float or int): Range of rotate to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-anglez, +anglez). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int): Range of shear to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-shear, +shear). Set to 0 to desactivate shear.
        interpolation ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
        fill_color (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(
        self,
        anglez,
        translate=(0, 0),
        scale=1,
        shear=0,
        fill_color=(0, 0, 0),
    ):
        self.anglez = anglez
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill_color = fill_color

    def image_transform(self, metadata):
        return affine6(
            metadata,
            self.anglez,
            self.translate,
            self.scale,
            self.shear,
            self.fill_color,
        )


class RandomAffine6(BaseRandomTransform):

    """Random affine transformation of the image keeping center invariant

    Args:
        anglez (sequence or float or int): Range of rotate to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-anglez, +anglez). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int): Range of shear to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-shear, +shear). Set to 0 to desactivate shear.
        interpolation ({NEAREST, BILINEAR, BICUBIC}, optional): An optional resampling filter.
        fill_color (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    
    def __init__(
        self,
        anglez=0,
        translate=(0, 0),
        scale=(1, 1),
        shear=1,
        fill_color=(0, 0, 0),
        prob=0.5,
    ):
        if isinstance(anglez, numbers.Number):
            if anglez < 0:
                raise ValueError("If anglez is a single number, it must be positive.")
            self.anglez = (-anglez, anglez)
        else:
            assert isinstance(anglez, (tuple, list)) and len(anglez) == 2, \
                "anglez should be a list or tuple and it must be of length 2."
            self.anglez = anglez

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

        if isinstance(shear, numbers.Number):
            if shear < 0:
                raise ValueError("If shear is a single number, it must be positive.")
            self.shear = (-shear, shear)
        else:
            assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                "shear should be a list or tuple and it must be of length 2."
            self.shear = shear
            
        self.fill_color = fill_color
        self.prob = prob

    @staticmethod
    def get_params(
        img_size,
        anglez_range=(0, 0),
        translate=(0, 0),
        scale_ranges=(1, 1),
        shear_range=(0, 0),
    ):
        angle = random.uniform(anglez_range[0], anglez_range[1])
        shear = random.uniform(shear_range[0], shear_range[1])

        max_dx = translate[0] * img_size[1]
        max_dy = translate[1] * img_size[0]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))

        scale = (random.uniform(1 / scale_ranges[0], scale_ranges[0]),
                 random.uniform(1 / scale_ranges[1], scale_ranges[1]))
        return angle, translations, scale, shear

    def image_transform(self, metadata):
        focus_image = get_focus_image_from_metadata(metadata)
        
        ret = self.get_params(
            self.anglez,
            self.translate,
            self.scale,
            self.shear,
            focus_image.shape,
        )
        return affine6(
            metadata,
            *ret,
            fill_color=self.fill_color,
        )
    