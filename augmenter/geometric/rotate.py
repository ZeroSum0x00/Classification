import cv2
import copy
import math
import random
import numbers
import numpy as np

from .resize import INTER_MODE
from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.bbox_processing import coordinates_converter



def transform_points(points, M):
    """
    Apply affine transform `M` (2x3) to a set of points.

    Args:
        points: np.ndarray, shape can be
            - (N, 2): single list of points
            - (N, K, 2): group of K points per item
            - (N, K, 3): like (x, y, z), apply only to x, y
        M: affine matrix of shape (2, 3)

    Returns:
        transformed points, with same shape as input
    """
    points = np.asarray(points)
    orig_shape = points.shape

    if points.ndim == 2 and points.shape[1] == 2:
        # shape (N, 2)
        pts = np.concatenate([points, np.ones((len(points), 1))], axis=-1)
        return (M @ pts.T).T

    elif points.ndim == 3 and points.shape[2] in [2, 3]:
        pts_xy = points[..., :2].reshape(-1, 2)  # (N*K, 2)
        ones = np.ones((pts_xy.shape[0], 1), dtype=pts_xy.dtype)
        pts_aug = np.concatenate([pts_xy, ones], axis=1)  # (N*K, 3)
        transformed_xy = (M @ pts_aug.T).T  # (N*K, 2)
        transformed_xy = transformed_xy.reshape(*points.shape[:-1], 2)

        if points.shape[2] == 3:
            out = points.copy()
            out[..., :2] = transformed_xy
            return out
        else:
            return transformed_xy
    else:
        raise ValueError(f"Unsupported point shape {points.shape}")


def rotate(
    metadata,
    angle,
    expand=False,
    center=None,
    fill_color=None,
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
        rank_size = 2
        gray_scale = True
    elif focus_image.ndim == 3:
        height, width, _ = focus_image.shape
        rank_size = 3
        gray_scale = False
    else:
        raise ValueError(f"Unsupported image shape: {focus_image.shape}")

    point = center or (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=1)

    if fill_color:
        if isinstance(fill_color, int):
            color = [random.randint(0, 255) if fill_color == -1 else fill_color] * rank_size
        else:
            color = fill_color
    else:
        color = [0] * rank_size

    new_w, new_h = width, height
    if expand:
        corners = np.array([
            [0, 0, 1],
            [width - 1, 0, 1],
            [width - 1, height - 1, 1],
            [0, height - 1, 1]
        ])
        transformed = np.dot(M, corners.T).T
        min_x, min_y = np.min(transformed[:, 0]), np.min(transformed[:, 1])
        max_x, max_y = np.max(transformed[:, 0]), np.max(transformed[:, 1])
        new_w = int(np.ceil(max_x - min_x))
        new_h = int(np.ceil(max_y - min_y))
        M[0, 2] += (new_w - width) / 2
        M[1, 2] += (new_h - height) / 2

    if metadata_check:
        if image_data is not None:
            img = image_data
            img = cv2.warpAffine(
                img, M, (new_w, new_h),
                flags=INTER_MODE[interpolation],
                borderValue=color
            )
            clone_data["image"] = img

        for key, array in auxi_image_data.items():
            if not isinstance(array, np.ndarray):
                continue
                
            clone_data["auxiliary_images"][key] = cv2.warpAffine(
                array, M, (new_w, new_h),
                flags=INTER_MODE[interpolation],
                borderValue=color
            )

        for key, array in masks_data.items():
            if not isinstance(array, dict):
                continue

            if isinstance(array, np.ndarray):
                clone_data["masks"][key] = cv2.warpAffine(array, M, (new_w, new_h), flags=INTER_MODE["NEAREST"], borderValue=0)
            elif isinstance(array, (list, tuple)):
                rotated = [cv2.warpAffine(v, M, (new_w, new_h), flags=INTER_MODE["NEAREST"], borderValue=0) for v in array]
                clone_data["masks"][key] = np.array(rotated)

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
    
                rotated_boxes = []
                for box in boxes:
                    x1, y1, x2, y2, lb = box
                    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    rotated = transform_points(corners, M)
                    x_min, y_min = rotated.min(axis=0)
                    x_max, y_max = rotated.max(axis=0)
                    rotated_boxes.append([x_min, y_min, x_max, y_max, lb])
                boxes = np.array(rotated_boxes)
    
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
            
            n = landmarks.shape[0]
            landmarks = transform_points(landmarks, M)
    
            if clip_out_range:
                landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, width - 1)
                landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, height - 1)
    
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        return cv2.warpAffine(
            image_data, M, (new_w, new_h),
            flags=INTER_MODE[interpolation],
            borderValue=color
        )


class Rotate(BaseTransform):

    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(
        self,
        angle,
        expand=False,
        center=None,
        fill_color=None,
    ):
        self.angle = angle
        self.expand = expand
        self.center = center
        self.fill_color = fill_color

    def image_transform(self, metadata):
        return rotate(
            metadata,
            angle=self.angle,
            expand=self.expand,
            center=self.center,
            fill_color=self.fill_color,
        )


class RandomRotate(BaseRandomTransform):

    """Rotate the image in range angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(
        self,
        angle,
        expand=False,
        center=None,
        fill_color=None,
        prob=0.5,
    ):
        if isinstance(angle, numbers.Number):
            if angle < 0:
                raise ValueError("If angle is a single number, it must be positive.")
            self.angle = (-angle, angle)
        else:
            if len(angle) != 2:
                raise ValueError("If angle is a sequence, it must be of len 2.")
            self.angle = angle

        self.expand = expand
        self.center = center
        self.fill_color = fill_color
        self.prob = prob

    @staticmethod
    def get_params(angle):
        return random.uniform(angle[0], angle[1])

    def image_transform(self, metadata):
        angle = self.get_params(self.angle)
        return rotate(
            metadata,
            angle=angle,
            expand=self.expand,
            center=self.center,
            fill_color=self.fill_color,
        )
    